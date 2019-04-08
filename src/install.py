#!/usr/bin/env python3
# Licensed under the MIT License
# This script installs libraries and dependencies of AutCar

def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime CI build driver.", usage='')

    parser.add_argument("--update", action='store_true', help="Update makefiles.")
    parser.add_argument("--build", action='store_true', help="Build.")

    return parser.parse_args()

def is_windows():
    return sys.platform.startswith("win")

def is_ubuntu_1604():
    return platform.linux_distribution()[0] == 'Ubuntu' and platform.linux_distribution()[1] == '16.04

def main():
    args = parse_arguments()

    a = 1