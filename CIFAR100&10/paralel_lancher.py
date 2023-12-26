import subprocess
from multiprocessing import Process
from time import time
import sys
def run_script(script_path, output_file):
    # print(12)
    # Run the Python script and capture the printed output
    # print(script_path)
    result = subprocess.run("python "+ script_path, shell=True, capture_output=True, text=True)

    # Get the printed output from the completed process
    printed_output = result.stdout
    # print(printed_output)
    # Save the printed output to a file
    with open(output_file, 'w') as f:
        f.write(printed_output)

# List of scripts and output file names you want to run in parallel
scripts_to_run = [
    {'script_path': r'dawn.py --world_size=2 --rank=1'},
    {'script_path': r'dawn.py --world_size=2 --rank=0'}
    # Add more scripts and output files as needed
]
test_nb=''
types=['layerwise']
networks=["Resnet9", "Alexnet"]
compress=['Topk_layer', 'Topk_level', 'Randomk_level', 'Randomk_layer']
# compress=['Randomk_adaptive', 'Topk_adaptive']

params=['22', '12', '02', '21', '11', '01', '20', '10', '00']
configs={
        "layerwise": ['Randomk_layer', 'Randomk_level', 'Topk_level', 'Topk_layer']}

# params=['000', '001', '002', '003', '010', '011', '012', '020', '022', '030', '033',
#         '100', '101', '110', '111', '123', '131', 
#         '200', '202', '210', '220', '222', '230', '232', 
#         '300', '301', '303', '313', '321', '323', '330', '333']
# networks=["Resnet9"]
# compress=['Topk_layer', 'Topk_level', 'Randomk_level', 'Randomk_layer']
# params=['220', '330',
#         '011', '022', '033',
#         '101', '202', '303',
#         '131', '313', '323', '232',
#         '012', '123', '230', '301', '321', '012', '210']
s0=time()
# Create and start a process for each script
for test in range(5):
    for typ in configs.keys():
        for network in networks:
            for method in configs[typ]:
                for parameter in params:
                    
                    processes = []
                    s1=time()
                    for indx, script_info in enumerate(scripts_to_run):
                        outputfile='logs13/test'+str(test)+'_'+typ+'_'+network+'_'+method+'_'+parameter+'_node'+str(indx)+'.txt'
                        process = Process(target=run_script, args=(script_info['script_path']+' --compress='+typ+' --network='+network + " --method="+method+" --extras="+parameter, outputfile))
                        processes.append(process)
                        process.start()

                    # Wait for all processes to complete
                    for process in processes:
                        process.join()
                    print("test", test," with",typ, network, method, parameter, time()-s1)

print("All scripts have been executed in parallel.",time()-s0)
