az login
az webapp up --name test-dash-v5 --runtime PYTHON:3.9 --resource-group Alpaca --sku FREE
                                                                                  # B1 also cheap

az webapp deployment user set --user-name K... --password T..1_
az webapp create --resource-group Alpaca --plan AlpacaMarketDashboardServicePlan --name test-dash-v5 --runtime PYTHON:3.9 --deployment-local-git
az webapp config appsettings set --name test-dash-v6 --resource-group Alpaca --settings DEPLOYMENT_BRANCH='main'
git remote add azure https://Kalmar@test-dash-v6.scm.azurewebsites.net/test-dash-v6.git
# "deploymentLocalGitUrl": "https://Kalmar@test-dash-v6.scm.azurewebsites.net/test-dash-v6.git"
git push azure main



az appservice list-locations --sku FREE # To see all supported locations
az webapp list-runtimes # All supported runtimes

az group create --name Alpaca --location "West Europe" # Create a resource group
az appservice plan create --name AlpacaMarketDashboardServicePlan --resource-group Alpaca --sku FREE --is-linux

# If the URL of azure git changed:
git remote -v
git remote remove azure

# After updating the app:
git commit -am "updated output"
git push azure main
