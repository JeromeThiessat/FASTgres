#!/usr/bin/sudo bash

sudo service postgresql stop
sync
#sudo echo 3 > /proc/sys/vm/drop_caches
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo service postgresql start