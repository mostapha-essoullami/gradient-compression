import subprocess
from multiprocessing import Process
from time import time
import os
import argparse
from datetime import datetime

def run_script(script_path, output_file):
    # print(12)
    # Run the Python script and capture the printed output
    print("python "+ script_path)
    result = subprocess.run("python "+ script_path, shell=True, capture_output=True, text=True)

    # Get the printed output from the completed process
    # printed_output = result.stdout+result.stderr
    # print(printed_output)
    # Save the printed output to a file
    # with open(output_file, 'w') as f:
    #     f.write(printed_output)

all_logs=['resnet9_cifar10_test_'+datetime.today().strftime('%Y-%m-%d')]
compress='layerwise'
master='tcp://localhost:2229'
networks=['Resnet9']
# List of scripts and output file names you want to run in parallel
scripts_to_run = [
    {'script_path': r'dawn.py --world_size=1 --rank=0  --gpu=3'+' --master_address='+master},
    # {'script_path': r'dawn.py --world_size=2 --rank=1 --qstates=10 --gpu=3'+' --master_address='+master+' --compress='+compress }
    # Add more scripts and output files as needed
]
test_nb=''

# methods=['Topk-adaptative', 'Topk-layer', 'Topk-level']
methods=['Topk_level_adaptative']
params=[0.1, 0.01, 0.001, 0.0001]

params.reverse()
all_logs.reverse()
# methods.reverse()
s0=time()
# Create and start a process for each script
for logs in all_logs:
    if not os.path.exists(logs):
        os.mkdir(logs)
for test in range(15,17):
        for network in networks:
            for method in methods:
                picking=0
                for param in params:
                    used_param=param
                    processes = []
                    s1=time()
                    for indx, script_info in enumerate(scripts_to_run):
                        outputfile=logs+'/test'+str(test)+'_'+network+'_'+method.replace('_','-')+'_'+str(used_param)+'_node'+str(indx)+'.txt'
                        # add_args=' --network='+network + " --method="+method +" --ratio="+str(test) +" --extras="+used_param
                        # add_args=' --network='+network + " --method="+method +" --k_min0="+str(g0) + ' --k_max0='+str(g1)+ ' --memory='+str(memory)
                        process = Process(target=run_script, args=(script_info['script_path']+ " --ratio="+str(test) +" --learning_rate="+str(param) , outputfile))
                        processes.append(process)
                        process.start()

                    # Wait for all processes to complete
                    for process in processes:
                        process.join()
                    print("done with test", test, network, compress, method, used_param, time()-s1, datetime.now().strftime('%H:%M:%S'))

print("All scripts have been executed in parallel.",time()-s0)

#python paralel_lancher_all.py  --network=Alexnet --compress=layerwise --method=Topk_layer_adaptative
