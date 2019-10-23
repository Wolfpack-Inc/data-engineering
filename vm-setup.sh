# Adjusted from https://github.com/ekhtiar/streaming-data-pipeline/blob/master/install.sh

# Install packages to allow apt to use a repository
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

# Add Docker’s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Set up the stable repository
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Update the apt package index.
sudo apt-get -y update

# Install docker-ce
sudo apt-get install -y docker-ce

# Check the current release and if necessary update
sudo curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose

# Set the permissions
sudo chmod +x /usr/local/bin/docker-compose

# Docker compose up
docker-compose up