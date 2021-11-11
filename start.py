import sys
import subprocess
import argparse
from pathlib import Path

def main(args):
    host = args.host
    port = args.port
    tm_port = args.tm_port
    cam_setup = args.cam_setup
    folder = args.folder 

    # Copy over the file sets.txt, to make it easier for methods using UTOCS to use it 
    sets_text = Path('sets.txt').read_text()
    new_sets_file = Path(folder) / 'sets.txt'
    new_sets_file.write_text(sets_text)

    scenario_string = args.range
    start, stop = [int(v) for v in scenario_string.split('-')]

    for s_num in range(start, stop+1):
        command = [sys.executable, 'utocs.py', '--host', host, '--port', port, '--tm_port', tm_port, '--cam_setup', cam_setup, '--folder', folder, '--scenario_number', str(s_num)]
        print(command)
        p = subprocess.Popen(command, shell=False)
        result = p.wait()

        if result != 0:
            print("IMPORTANT! You need to reset the CARLA server to ensure the resulting videos will be valid")

            raise ValueError(f"Failure! Error code {result}. Error messages: {p.stderr}")
            
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--host", default='127.0.0.1', help="IP of the Carla server host")
    args.add_argument("--port", default='2000', help="Port to connect to the server")
    args.add_argument("--tm_port", default='8000', help="Traffic Manager communications port")
    args.add_argument("--cam_setup", default='0,0,0', help="List of camera offsets. To make a stereo camera setup with 0.5 metres distance, do '0,0,0;0.5,0,0'. Each tuple contains x, y and z distances from the default camera position. The camera is always facing in positive z direction. Separate by semicolons (note that you probably need quotes around this argument).")
    args.add_argument("--folder", default="./output", type=str, help="Folder to store output (default is './output')")
    args.add_argument("--range", default='0-23', type=str, help="Which scenarios to run, as two integers separated by a dash, like '0-23' (inclusive)")

    args = args.parse_args()
    main(args)
