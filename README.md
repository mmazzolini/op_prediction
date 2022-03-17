# Operational prediction with SVR

 operational discharge prediction for the ADO project.
 to create the needed environment type in conda: 
 
 $conda create --name myenv --file spec-file.txt

Input data from ZAMG is available on the EURAC ADO project repository, that could be mounted at letter Z:

The models are available for ADO partners at the following link: https://scientificnet.sharepoint.com/:f:/r/sites/adoproject/Shared%20Documents/T2/Runoff_prediction/models?csf=1&web=1&e=4A4fdX

for the prediction with a lead time also the climatology is needed. That is available for ADO partners at: https://scientificnet.sharepoint.com/:f:/r/sites/adoproject/Shared%20Documents/T2/Runoff_prediction/climatology?csf=1&web=1&e=7gCVzp


The results are inserted in a DB under the scheme named ML_discharge, instructions for accessing it are available at the link https://edp-portal.eurac.edu/cdb_doc/ado/
