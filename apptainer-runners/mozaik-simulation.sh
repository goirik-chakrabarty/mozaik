cd /project
rm -r SelfSustainedPushPull_test:test8_____

# python run_parameter_search.py run_spont.py nest param_MSA/defaults 
# python run_spont.py nest 2 param_MSA/defaults 'test'
# echo "Waiting for debugger attach..."
# python -m debugpy --listen 0.0.0.0:5678 --wait-for-client run.py nest 2 param_MSA/defaults 'test:3'
python -u run.py nest 8 param_MSA/defaults 'test:test8'