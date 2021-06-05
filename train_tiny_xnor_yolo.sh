knockknock slack \
	--webhook-url https://hooks.slack.com/services/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 \
	--channel dev_github_action_noti \	
	python3 main.py train --dataset-config=conf/data/data.yml --model-config=conf/model/xnor-tiny-yolo.yml --runner-config=conf/training/xnor-tiny-yolo_training.yml
