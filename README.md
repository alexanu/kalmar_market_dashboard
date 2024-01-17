# Alpaca Market Dashboard hosted on Azure Web App
The dashboard is created with Dash

Data sources: Alpaca and FMP

To run locally: open folder in terminal => `python app.py`

## Creating variables
Using CMD

     set AZ_LOCATION='eastus'
     set AZ_RESOURCE_GROUP_NAME='XXXXX'
     set APP_SERVICE_PLAN_NAME='XXXXXXXXXXX'
     set PLAN='B1'
     set RUNTIME='PYTHON:3.9'
     set APP_SERVICE_NAME='XXXXXX'

## Quickly deploy Dash file as Azure Web App
     az login
     az webapp up 
          --name $APP_SERVICE_NAME 
          --runtime $RUNTIME 
          --resource-group $RESOURCE_GROUP_NAME 
          --sku $PLAN
Set the appsettings for the API key (the command is in .gitignore file)


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