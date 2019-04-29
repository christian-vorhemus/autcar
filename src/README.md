## Hardware Assembly



## Software

### 1) Configure your Raspberry Pi

If you already have a Rasperry Pi 3 with the newest Raspbian Lite up and running, you can skip this section.

1) Download Raspbian Lite from [here](https://www.raspberrypi.org/downloads/raspbian/)
2) Download a tool to write disk images to a SD card. On Windows, you can use [Win32 Disk Imager](https://www.heise.de/download/product/win32-disk-imager-92033)
3) Download a SSH client, for example [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
4) Plug a SD card to your computer, start your disk image writer and write the Raspian .img file to the card
5) Open the /boot partition of the SD card, and create a new file named "wpa_supplicant.conf"
6) Add the following content to the file but replace <YOUR-WIFI-SSID> and <YOUR-WIFI-KEY> with the SSID (name) and password of your WLAN access point.
```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=AT

network={
    ssid="<YOUR-WIFI-SSID>"
    psk="<YOUR-WIFI-KEY>"
    key_mgmt=WPA-PSK
}
```
6) Create an empty file "ssh" (no file ending!) on /boot partition to enable ssh
7) Start your Raspberry Pi and search on your WLAN access point which IP address your Raspberry Pi is using. Most access points provide a web interface you can use to find the IP address, however, the address of the web interface differrs from router to router. If you don't know it, open a browser and try 192.168.1.1, 10.0.0.1 or 10.0.0.138 to open the web interface and note down the IP address of your Raspberry.
8) Open PuTTY, enter the IP address of your Raspberry Pi and click "Open". The default credentials are
```
Username: pi
Password: raspberry
```

It is recommended to change your password. To do so, enter 
```
passwd
```
and change the default password.