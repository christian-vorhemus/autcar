import subprocess
import argparse
import os
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description="AutCar install script", usage='')

    parser.add_argument("--prevent-reboot", action='store_true', help="Prevent rebooting after install script finished")
    parser.add_argument("--prevent-update", action='store_true', help="Don't run apt-get update + upgrade")
    parser.add_argument("--prevent-swap", action='store_true', help="Don't create a swap file")

    return parser.parse_args()

def reboot():
	subprocess.check_call("sudo reboot", shell=True)

def install_onnxruntime():
    from pip import pep425tags
    supported_tags = pep425tags.supported_tags
    paths = glob.glob("tools/*.whl")
    filepath = ""

    for supported_tag in supported_tags:
        for tag in supported_tag:
            if(tag.startswith("linux")):
                for path in paths:
                    if(tag in path):
                        filepath = path

    if(filepath == ""):
        raise Exception("No supported wheel file found in /tools folder")
    else:
        subprocess.check_call("python3 -m pip install "+filepath, shell=True)

def create_swap():
    if(os.path.isfile('./swapfile') is False):
        subprocess.check_call("sudo dd if=/dev/zero of=swapfile bs=1M count=2500", shell=True)
    try:
        subprocess.check_call("sudo mkswap swapfile", shell=True)
        subprocess.check_call("sudo swapon swapfile", shell=True)
    except Exception as e:
        return

def install_pip():
    subprocess.check_call("sudo apt-get -y install python3-pip", shell=True)

def enable_camera():
    subprocess.check_call("sudo raspi-config nonint do_camera 0", shell=True)

def update():
    subprocess.check_call("sudo apt-get update && sudo apt-get -y upgrade", shell=True)

def install_git():
    subprocess.check_call("sudo apt-get -y install git", shell=True)

def install_gpio():
    subprocess.check_call("sudo apt-get -y install python3-rpi.gpio", shell=True)

def install_numpy():
    subprocess.check_call("sudo apt-get -y install python3-dev python3-numpy", shell=True)

def uninstall_old_numpy():
    subprocess.check_call("python3 -m pip uninstall numpy --yes", shell=True)

def install_pillow():
    subprocess.check_call("sudo apt-get -y install libjpeg-dev zlib1g-dev", shell=True)
    subprocess.check_call("python3 -m pip install Pillow", shell=True)

def install_imageops():
    subprocess.check_call("sudo apt-get -y install python-imaging", shell=True)

def install_libqt4():
    subprocess.check_call("sudo apt-get -y install libqt4-test", shell=True)

def install_opencv():
    subprocess.check_call("sudo apt-get -y install libcblas-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libilmbase-dev openexr libgstreamer1.0-0 libavcodec-dev libavformat-dev libswscale-dev", shell=True)
    subprocess.check_call("python3 -m pip install opencv-python", shell=True)

def main():
    args = parse_arguments()

    if(args.prevent_update == False):
        print("Update Raspberry...")
        update()
        print("Update finished")

    if(args.prevent_swap == False):
        print("Preparing swapfile")
        create_swap()

    print("Enable camera...")
    enable_camera()
    print("Camera enabled. Reboot required")

    print("Install pip...")
    install_pip()
    print("Pip installed")

    print("Install GPIO connectors...")
    install_gpio()
    print("GPIO installed")

    print("Install numpy...")
    install_numpy()
    print("Numpy installed")

    print("Install Pillow...")
    install_pillow()
    print("Pillow installed")

    print("Install ImageOps...")
    install_imageops()
    print("ImageOps installed")

    print("Install libqt4...")
    install_libqt4()
    print("Libqt4 installed")

    print("Install OpenCV...")
    install_opencv()
    print("OpenCV installed")

    print("Install ONNXRuntime")
    install_onnxruntime()
    print("ONNXRuntime installed")

    print("Uninstall old numpy version...")
    uninstall_old_numpy()
    print("Old numpy uninstalled")

    print("Sucessfully installed AutCar platform! Reboot required.")

    if(args.prevent_reboot == False):
        print("Rebooting now...")
        reboot()

if __name__ == '__main__':
    main()
