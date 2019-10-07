1- Execute these commands:

    $ sudo cp overclock-jetson.sh /usr/bin/overclock-jetson.sh
    $ sudo chmod +x /usr/bin/overclock-jetson.sh

2- Create a service file (overclock-jetson.service):

    [Unit]
    Description=Overclock jetson.

    [Service]
    Type=idle
    ExecStart=/bin/bash /usr/bin/overclock-jetson.sh

    [Install]
    WantedBy=default.target

3- Execute these commands:

    $ sudo cp overclock-jetson.service /etc/systemd/system/overclock-jetson.service
    $ sudo chmod 644 /etc/systemd/system/overclock-jetson.service

4- Test your service:

    $ sudo systemctl start overclock-jetson.service
    $ sudo systemctl status overclock-jetson.service

5- If everything is ok, deploy the service:

    $ sudo systemctl enable overclock-jetson.service
    $ sudo reboot