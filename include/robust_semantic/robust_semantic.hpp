/***********************************************************
 *
 * @file: base_planner.h
 * @breif: Contains common/commonly used nodes data strcutre
 * @author: David Morilla-Cabello
 * @update: TODO
 * @version: 1.0
 *
 * Copyright (c) 2023， David Morilla-Cabello
 * All rights reserved.
 * --------------------------------------------------------
 *
 **********************************************************/
#ifndef ROBUST_SEMANTIC_H_
#define ROBUST_SEMANTIC_H_
#include <unordered_map>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <ipp_tools/planners/basics/poses_file_planner.h>
#include <ipp_tools/planners/bridges/airsim_planner_bridge.h>

namespace robust_semantic {

// Alias
// typedef ipp_tools::planners::basics::PosesFilePlanner PosesFilePlanner;
// typedef ipp_tools::planners::bridges::AirsimPlannerBridge AirsimPlannerBridge;

enum class DataSource {
  SIMULATOR, // The node uses the planner to move the camera
  DATASET, // The node commands the movement to the sensor as the dataset
           // is better managed in Python
};

class RobustSemanticNode {
 public:
  RobustSemanticNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
  ~RobustSemanticNode();

 private:
  // Members
  std::shared_ptr<ipp_tools::planners::basics::PosesFilePlanner> planner_;
  int curr_pose_idx_;
  std::shared_ptr<ipp_tools::planners::bridges::AirsimPlannerBridge> bridge_;
  
  // ROS related members
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // Subscribers
  ros::Subscriber sub_pcd_;

  // Services
  ros::ServiceClient srv_client_sensor_;
  ros::ServiceClient srv_client_move_sensor_;

  // Controller

  DataSource p_data_source_;
  std::string p_poses_path_;
  std::string p_poses_file_;
  std::vector<Eigen::Affine3f> poses_;

  // Methods

  void rosInit();

  // Callbacks

  void pcdCallback(sensor_msgs::PointCloud2ConstPtr const &msg);

  void sendCaptureRequest();
  void sendMoveRequest();
};

}  // namespace robust_semantic

#endif // ROBUST_SEMANTIC_H_