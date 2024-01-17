#include "ros/ros.h"

#include "robust_semantic/robust_semantic.hpp"

int main(int argc, char **argv) {
    ros::init(argc, argv, "robust_semantic_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    ROS_INFO("Starting robust_semantic_node...");
    robust_semantic::RobustSemanticNode robust_semantic_node(nh, nh_private);

    while (ros::ok()) {
        ros::spinOnce();
    }
    return 0;
}
