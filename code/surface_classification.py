#!/usr/bin/env python
import rospy, rosbag
from sensor_msgs.msg import LaserScan, Imu, Joy, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16, Float32

from cv_bridge import CvBridge, CvBridgeError
import cv2

import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# Global vars
step_size = 40 # how many imu measurements before reading
pub = rospy.Publisher('surface_detection', Int16, queue_size=10) # initializing the publisher for this node
bumpiness_pub = rospy.Publisher('surface_bumpiness', Float32, queue_size=10)
camera_pub = rospy.Publisher('crop_image',Image,queue_size=10)

# DBSCAN vars
eps = .15 # distance from center to create cluster
#min_samples = 15 # number of samples needed to create new cluster
min_samples = 10

# Scaling parameters
max_int = 950
min_int = 500
max_rng = .03
min_rng = 0
max_z_accel = 1.48
min_z_accel = .0069
max_z_angular = .65
min_z_angular = -1.2
max_pixel = 100
min_pixel = 0

def normalize(value, maximum, minimum):
	res = (value - minimum) / (maximum - minimum)
	if res < 0:
		res = 0
	elif res > 1:
		res = 1
	return res

class SurfaceMonitor:
	curr_acc = []
	curr_z_ang = []
	curr_lidar_int = []
	curr_lidar_range = []
	curr_x_vel = []
	curr_edges = []
	overall_vector = []
	recent_vector = []
	test_vector = []
	count = 0

	# State params
	state = 0
	state0_exit_count = 25 # The number of times a cluster is read as the same value before switching out of state 0
	neg_one_count = 0 # Tracks number of times -1 has been read in a row
	neg_one_thresh = 5 # Threshold of -1 readings in a row before moving state
	state2_exit_count = 15 # Number of times non -1 has to be read in a row before saving test vector
	test_count_limit = 5 # Limit of test runs before checking to throw out overall_vector
	perc_neg_one = .75 # Percent limit of negative ones in recent readings to continue using test vector
	test_count = 0 # Keeps track of number of runs through state 2

	def imu_callback(self, data):
		# z accel
		self.curr_acc.append(data.linear_acceleration.z)

		# z angular
		self.curr_z_ang.append(data.angular_velocity.z)

		if len(self.curr_acc) >= step_size:
			self.check_surface()
			self.reset_curr_arrs()

	def camera_callback(self, data):
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(data,"rgba8")

		width, height, channels = cv_image.shape
		#crop_image = cv_image
		crop_image = cv_image[int(height*0.5):height, 0:width]
		edges = cv2.Canny(crop_image,100,200)

		camera_pub.publish(bridge.cv2_to_imgmsg(edges,"mono8"))

		m_edges_rows = np.mean(edges,axis=0)
		m_edges = np.mean(m_edges_rows, axis=0)

		self.curr_edges.append(m_edges)

	def lidar_callback(self, data):
		# handle intensity
		self.curr_lidar_int.append(sum(data.intensities)/len(data.intensities))

		# handle range
		z_displacements = []
		for i in range(358):
			z_displacements.append(math.sin(0.6108652)*math.cos(-0.781035+ .004363*i)*data.ranges[i])
		self.curr_lidar_range.append(np.std(z_displacements))

		#if len(self.curr_lidar_range) >= step_size:
		#	self.check_surface()
		#	self.reset_curr_arrs()


	def check_surface(self):
		if len(self.curr_lidar_int) == 0:
			print("Lidar not working")
			return

		# Modify measurement arrays so they're all a single value
		accel = np.std(self.curr_acc)
		angular = sum(self.curr_z_ang)/float(len(self.curr_z_ang))
		intensity = sum(self.curr_lidar_int)/float(len(self.curr_lidar_int))
		rng = sum(self.curr_lidar_range)/float(len(self.curr_lidar_range))
		vel = 0
		if len(self.curr_x_vel) > 0:
			vel = sum(self.curr_x_vel)/len(self.curr_x_vel)
		edge = np.mean(self.curr_edges)

		#print(rng)

		# Publish range data
		bumpiness_pub.publish(rng)

		# Perform normalization
		accel = normalize(accel, max_z_accel, min_z_accel)
		angular = normalize(angular, max_z_angular, min_z_angular)
		intensity = normalize(intensity, max_int, min_int)
		rng = normalize(rng, max_rng, min_rng)
		edge = normalize(edge, max_pixel,min_pixel)
		#print(rng)
		#print("----------------------")


		# Create vector for past step
		print(str(intensity) + "," + str(rng) + "," + str(edge))
		self.recent_vector = [[intensity, rng, edge]]

		# If state 0, append to overall vector normally and perform DBSCAN after 30 measurements
		if self.state == 0:
			self.overall_vector += self.recent_vector
			self.perform_dbscan(self.overall_vector)

		# If state 1, perform DBSCAN by adding recent_vector onto overall_vector without actually updating it
		elif self.state == 1:
			self.perform_dbscan(self.overall_vector + self.recent_vector)

		# If state 2, use test vector
		elif self.state == 2:
			# Update test vector to overall vector if behind it
			if len(self.test_vector) < len(self.overall_vector):
				self.test_vector = list(self.overall_vector)
			self.test_vector += self.recent_vector # add most recent reading
			self.perform_dbscan(self.test_vector) # perform DBSCAN

	def perform_dbscan(self, vector):
		db = DBSCAN(eps=eps, min_samples=min_samples).fit(vector) # Create DBSCAN model and fit it to the vector
		self.count += 1

		# Status printing
		#print("Run: " + str(self.count))
		#print("Current state: " + str(self.state))
		#print(db.labels_)
		#print(str(self.count) + "," + str(db.labels_[-1])) # print out the classified labels for the vector
		#print("-----------------------------------------------------------")
		labels = db.labels_

		# Publish surface identification
		pub.publish(labels[-1])

		# HANDLE STATE MACHINE
		# If state 0, move to state 1 if 80% of last state0_exit_count # of readings are 0
		if self.state == 0:
			if list(labels[-self.state0_exit_count:]).count(0) / float(self.state0_exit_count) >= .8:
				self.overall_vector = [self.overall_vector[i] for i,x in enumerate(labels) if x != -1]
				self.state = 1

		# If state 1, move to state 2 if reads -1 neg_one_thresh # of times in a row
		elif self.state == 1:
			if labels[-1] == -1:
				self.neg_one_count += 1
			else:
				self.neg_one_count = 0
			if self.neg_one_count >= self.neg_one_thresh:
				self.state = 2
				self.test_count = 0
				self.neg_one_count = 0

		# If state 2, move to state 1 and update overall_vector if classifies to same cluster
		# same_reading_count # of times in a row. If it passes the test_count_limit before
		# this happens, dump test_vector and return to state 1 if not too many neg ones recently
		elif self.state == 2:
			self.test_count += 1
			prev = DBSCAN(eps=eps, min_samples=min_samples).fit(self.overall_vector).labels_
			prev_clusters = len(set([x for x in prev if x != -1]))
			test_clusters = len(set([x for x in labels if x != -1]))
			if test_clusters > prev_clusters and list(labels[-15:]).count(test_clusters-1) / 15.0 >= .8:
				self.overall_vector = [self.test_vector[i] for i,x in enumerate(labels) if x != -1]
				self.test_vector = []
				self.state = 1
			elif self.test_count > self.test_count_limit:
				if test_clusters > prev_clusters: # new cluster has been found
					subset = list(labels[-self.test_count_limit:])
					if (subset.count(-1) + subset.count(test_clusters-1)) / self.test_count_limit > self.perc_neg_one:
						self.test_count = 0
					else:
						self.test_vector = []
						self.state = 1
				else:
					perc_neg_one = list(labels[-self.test_count_limit:]).count(-1) / self.test_count_limit
					if perc_neg_one > self.perc_neg_one:
						self.test_count = 0
					#elif perc_neg_one == 0: # Controlled expansion of previous clusters
					#	self.overall_vector = [self.test_vector[i] for i,x in enumerate(labels) if x != -1]
					#	self.test_vector = []
					#	self.state = 1
					else:
						self.test_vector = []
						self.state = 1

	def reset_curr_arrs(self):
		self.curr_acc = []
		self.curr_lidar_int = []
		self.curr_lidar_range = []
		self.curr_x_vel = []
		self.curr_z_ang = []
		self.curr_edges = []

def main():
	monitor = SurfaceMonitor()
	rospy.init_node("surface_detection", anonymous=True)
	rospy.Subscriber("/back_scan", LaserScan, monitor.lidar_callback)
	rospy.Subscriber("/imu/data", Imu, monitor.imu_callback)
	rospy.Subscriber("/camera/rgb/image_raw", Image, monitor.camera_callback)
	while not rospy.is_shutdown():
		pass

if __name__ == "__main__":
	main()
