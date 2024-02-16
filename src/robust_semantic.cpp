#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <std_srvs/SetBool.h>

#include <robust_semantic/robust_semantic.hpp>

namespace robust_semantic {

std::unordered_map<std::string, DataSource> data_source_map = {
    {"SIMULATOR", DataSource::SIMULATOR},
    {"DATASET", DataSource::DATASET},
};

RobustSemanticNode::RobustSemanticNode(ros::NodeHandle &nh,
                                       ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private) {
    rosInit();

    ROS_INFO("RobustSemanticNode initialized");
    // Warm-up phase for the planner
    if (p_data_source_ == DataSource::SIMULATOR) {
        ROS_INFO("Initializing AirsimBridge");
        bridge_ = std::make_shared<ipp_tools::planners::bridges::AirsimPlannerBridge>();

        ROS_INFO("Initializing PosesPlanner");
        planner_ = std::make_shared<ipp_tools::planners::basics::PosesFilePlanner>();
        planner_->setup(p_poses_file_);
        // Get all the poses
        planner_->plan();
        poses_ = planner_->getPath();
        curr_pose_idx_ = 0;
        Eigen::Affine3f first_pose = poses_[curr_pose_idx_];
        // Send the first pose to the sensor
        bridge_->sendPose(first_pose);

    } 
    ros::Duration(10.0).sleep();
    ROS_INFO("Warm-up phase finished");
    // Send capture request
    sendCaptureRequest();
}

RobustSemanticNode::~RobustSemanticNode() {}

void RobustSemanticNode::rosInit() {
    std::string data_source_type;

    if (!nh_private_.getParam("data_source", data_source_type)) {
        ROS_WARN("data_source not set, using default: SIMULATOR");
        data_source_type = "SIMULATOR";
    }
    if (data_source_type == "SIMULATOR") {
        if (!nh_private_.getParam("poses_path",
                                  p_poses_path_))  // !THIS is currently public
        {
            ROS_WARN(
                "poses_path not set, using default: "
                "/tmp/semantic_mapping/poses");
            p_poses_path_ = "/tmp/semantic_mapping/poses";
        }
    }

    p_data_source_ = data_source_map[data_source_type];

    // Get poses file
    if (p_data_source_ == DataSource::SIMULATOR) {
        p_poses_file_ = p_poses_path_ + "run.txt";
        ROS_INFO("Run path: %s", p_poses_file_.c_str());
        // Check if file exists
        if (!std::filesystem::exists(p_poses_file_)) {
            // Critical error if file does not exist and exit
            ROS_ERROR("File %s does not exist", p_poses_file_.c_str());
            exit(1);
        }
    }

    // Subscribers

    sub_pcd_ =
        nh_.subscribe("point_cloud", 1, &RobustSemanticNode::pcdCallback, this);

    // Services
    srv_client_sensor_ = nh_.serviceClient<std_srvs::SetBool>("capture_data");
    srv_client_move_sensor_ =
        nh_.serviceClient<std_srvs::SetBool>("move");
}

void RobustSemanticNode::pcdCallback(
    sensor_msgs::PointCloud2ConstPtr const &msg) {
    // Received data from sensor, we can move to the next pose and send a
    // capture request
    // ROS_INFO("Received point cloud");

    // Move to next pose
    if (p_data_source_ == DataSource::SIMULATOR) {
        // Send the next pose to the sensor
        if (curr_pose_idx_ >= poses_.size()) {
            ROS_INFO("Reached limit of steps, shutting down");
            ros::shutdown();
            return;
        }
        Eigen::Affine3f next_pose = poses_[curr_pose_idx_++];
        bridge_->sendPose(next_pose);

        // Send capture request
        sendCaptureRequest();
    } else if (p_data_source_ == DataSource::DATASET) {
        // Send capture request
        sendMoveRequest();
        sendCaptureRequest();
    }
}

void RobustSemanticNode::sendCaptureRequest() {
    // Send a capture request through the service
    std_srvs::SetBool srv;
    srv.request.data = true;
    if (srv_client_sensor_.call(srv)) {
        // ROS_INFO("Capture request sent");
        if (p_data_source_ == DataSource::DATASET && !srv.response.success) {
            ROS_INFO("No more captures to be made, shutting down");
            ros::shutdown();
        }
    } else {
        ROS_ERROR("Failed to call service capture");
    }
}

void RobustSemanticNode::sendMoveRequest() {
    // Send a capture request through the service
    std_srvs::SetBool srv;
    srv.request.data = false;
    if (srv_client_move_sensor_.call(srv)) {
        // ROS_INFO("Capture request sent");
        if (!srv.response.success) {
            ROS_INFO("Problem detected when moving the sensor, shutting down");
            ros::shutdown();
        }
    } else {
        ROS_ERROR("Failed to call service move");
    }
}

}  // namespace robust_semantic