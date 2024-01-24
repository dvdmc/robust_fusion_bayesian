# In the folder /home/david/research/APbayDL/experiments there are maps. Under those maps there are folders with experiments
# This script evaluates all the experiments with the following command:
# rosrun semantic_mapping map_evaluator /home/david/research/APbayDL/experiments/*map_name*/*experiment_name*/
# For all maps and experiments

# Get the name of the maps
maps=$(ls /home/david/research/APbayDL/experiments)
# For each map
for map in $maps
do
    # Get the name of the experiments
    experiments=$(ls /home/david/research/APbayDL/experiments/$map)
    # Delete everything that is not a file
    # For each experiment
    for experiment in $experiments
    do
        # Evaluate the experiment
        echo "Evaluating $experiment"
        rosrun semantic_mapping map_evaluator /home/david/research/APbayDL/experiments/$map/$experiment/
    done
done