import subprocess
from multiprocessing import Process
from time import time
import os
import argparse

def run_script(script_path, output_file):
    # print(12)
    # Run the Python script and capture the printed output
    result = subprocess.run("python "+ script_path, shell=True, capture_output=True, text=True)

    # Get the printed output from the completed process
    printed_output = result.stdout+result.stderr
    # print(printed_output)
    # Save the printed output to a file
    with open(output_file, 'w') as f:
        f.write(printed_output)

# parser = argparse.ArgumentParser()
# parser.add_argument('--log_dir', type=str, default='logs')
# parser.add_argument('--network', type=str, default='Alexnet')
# parser.add_argument('--compress', '-r', type=str, default="layerwise")
# parser.add_argument('--method', '-w', type=str, default="Topk")
# parser.add_argument('--master_address', '-m', type=str, default='tcp://localhost:2222')

# args = parser.parse_args()

logs='adaptative-nomem'
compress='layerwise'
master='tcp://localhost:2232'
network='Resnet9'
# List of scripts and output file names you want to run in parallel
scripts_to_run = [
    {'script_path': r'dawn.py --world_size=2 --rank=1 --qstates=5'+' --master_address='+master+' --compress='+compress+ ' --network='+network },
    {'script_path': r'dawn.py --world_size=2 --rank=0 --qstates=5'+' --master_address='+master+' --compress='+compress+ ' --network='+network }
    # Add more scripts and output files as needed
]
test_nb=''

methods=['Topk_level_adaptative', 'Topk_layer_adaptative']

params=['01', '02', '10', '12', '20', '21', '22', '11' , '00']

s0=time()
# Create and start a process for each script
if not os.path.exists(logs):
    os.mkdir(logs)
for test in range(3):
    for method in methods:
        for parameter0 in params:
            for parameter1 in params:

                used_param=parameter0+parameter1
                processes = []
                s1=time()
                for indx, script_info in enumerate(scripts_to_run):
                    outputfile=logs+'/test'+str(test)+network+'_'+method.replace('_','-')+'_'+used_param+'_node'+str(indx)+'.txt'
                    process = Process(target=run_script, args=(script_info['script_path']+ " --method="+method +" --extras="+used_param, outputfile))
                    processes.append(process)
                    process.start()

                # Wait for all processes to complete
                for process in processes:
                    process.join()
                print("done with", network, compress, method, used_param, time()-s1)

print("All scripts have been executed in parallel.",time()-s0)

#python paralel_lancher_all.py  --network=Alexnet --compress=layerwise --method=Topk_layer_adaptative
