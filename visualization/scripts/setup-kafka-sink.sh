curl --header "Content-Type: application/json" --request PUT --data "$(cat /scripts/twitter-index.json)" http://elasticsearch:9200/twitter
curl --header "Content-Type: application/json" --request PUT --data "$(cat /scripts/crypto-index.json)" http://elasticsearch:9200/crypto
curl --header "Content-Type: application/json" --request PUT --data "$(cat /scripts/prediction-index.json)" http://elasticsearch:9200/prediction
curl --header "Content-Type: application/json" --request POST --data "$(cat /scripts/connector-setup.json)" http://localhost:8083/connectors