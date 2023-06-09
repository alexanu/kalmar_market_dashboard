# Alpaca Market Dashboard hosted on Azure Web App
The dashboard is created with Dash

Data sources: Alpaca and FMP

To run locally: open folder in terminal => `python app.py`

## Creating variables
Using CMD

     set AZ_LOCATION='eastus'
     set AZ_RESOURCE_GROUP_NAME='Alpaca'
     set APP_SERVICE_PLAN_NAME='XXXXXXXXXXX'
     set PLAN='B1'
     set RUNTIME='PYTHON:3.9'
     set APP_SERVICE_NAME='Alpaca-Market-Dashboard'

## Quickly deploy Dash file as Azure Web App
     az login
     az webapp up 
          --name $APP_SERVICE_NAME 
          --runtime $RUNTIME 
          --resource-group $RESOURCE_GROUP_NAME 
          --sku $PLAN
Set the appsettings for the API key (the command is in .gitignore file)

## Long way via git
### Prep
     az webapp deployment user set 
          --user-name K... 
          --password T..1_
     git init -b main
     git add .
     git commit -m "First Commit"


### Azure Web App creation
     az webapp create 
          --resource-group $RESOURCE_GROUP_NAME 
          --plan $APP_SERVICE_PLAN_NAME 
          --name $APP_SERVICE_NAME 
          --runtime $RUNTIME 
          --deployment-local-git

### Create Azure git
     az webapp config appsettings set 
          --name $APP_SERVICE_NAME 
          --resource-group $RESOURCE_GROUP_NAME 
          --settings DEPLOYMENT_BRANCH='main'
     git remote add azure https://<user>@<app name>.scm.azurewebsites.net/<app name>.git
The link for the above command is taken from the json respond to the previous command ("deploymentLocalGitUrl")
     
     git push azure main

### If the URL of azure git changed:
     git remote -v
     git remote remove azure

### After updating the app:
     git commit -am "updated output"
     git push azure main


# Help commands for Azure CLI
     az --version
     az appservice list-locations --sku FREE
     az webapp list-runtimes



`az storage blob list --account-name $STORAGE_NAME --container-name $CONTAINER_NAME --output table --auth-mode login`


# Git

## Init

    git init
    git add .
    git commit -m "initial commit"

    git remote add origin https...
    git branch -M main
    git push -u origin main

## Maintain

    git add -A
    git commit -m "Update"
    git push

## Update local content from Github: 

    git pull origin main

# Conda Virtual Environments for Windows:

    conda env list 
    conda create -n alpaca_env 
    conda install -n alpaca_env python #### or: conda install pip
    conda activate alpaca_env
    pip install -r requirements.txt
    conda deactivate
    conda env remove -n alpaca_env