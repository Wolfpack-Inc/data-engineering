# data-engineering

#### Google cloud installation steps
1. Create a new compute engine
2. Choose required settings. For OS-image choose `Ubuntu 18.04 LTS Minimal` and `15Gb` storage space
3. Create the compute engine
4. SSH into the vm when it is created
5. Run `sudo apt-get update`
6. Install git `sudo apt-get install git -y`
7. Clone the data-engineering repository `git clone https://github.com/Wolfpack-Inc/data-engineering.git`
8. Go into the repo `cd data-engineering`
9. Run the vm setup script `sh vm-setup.sh`
10. Open port 8080 and 5601 in the firewall (See opening ports)
11. Go to `<External-Ip>:8080` to check if everything is running
12. Setup the elasticsearch indices and the confluent connector

#### Setup confluent connector
1. Go to the root of the repository
2. Run `sh setup-kafka-connector.sh`

#### Opening a port in your firewall
1. Open the google shell
2. Run `gcloud compute firewall-rules create pyspark-notebook --allow tcp:8080 --source-ranges=0.0.0.0/0`
2. Run `gcloud compute firewall-rules create kibana --allow tcp:5601 --source-ranges=0.0.0.0/0`

#### Starting all services after restart
1. `cd data-engineering`
2. `sudo docker-compose up`
3. If you want to start as deamon `sudo docker-compose up -d`

#### Changed ingestion script
1. You will need to rebuild the ingestion image
2. Run `sudo docker-compose build`
3. Restart docker compose `sudo docker-compose up`
