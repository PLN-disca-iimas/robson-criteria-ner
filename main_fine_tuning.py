from utils import RobsonCriteriaNERModel
import logging
import time

if __name__ == "__main__":

    dataset_path = "data/anonymized_gold_dataset_3_sentences"
    batch_size = 2 #, 8,
    epochs = [3, 5, 10]
    models =[  
        {
            "model_name": "google-bert/bert-base-multilingual-cased",
            "short_model_name": "bert",
        },
        {
            "model_name": "BSC-TeMU/roberta-base-biomedical-es",
            "short_model_name": "roberta",
        },
        {
            "model_name": "FacebookAI/xlm-roberta-base",
            "short_model_name": "xlm_roberta",
        }
        
    ]

    base_path = "/media/discoexterno/orlando/anonymaized"
    # base_path = "./"

    rob_ner = RobsonCriteriaNERModel()
    rob_ner.set_base_path(base_path)
    rob_ner.load_dataset(dataset_path)

    for item in models:
        for i, epoch in enumerate(epochs):
            log_filename = f"snt_{item['short_model_name']}_{epoch}_{batch_size}.log"
            rob_ner.define_logger(log_filename)
            rob_ner.load_model_and_tokenizer(item['model_name'], item['short_model_name'])
            rob_ner.main(epoch, batch_size)
            time.sleep(30)  # sleep fo 30 seconds to avoid GPU memory issues
            for handler in list(rob_ner.logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    rob_ner.logger.removeHandler(handler)
                    handler.close()
                    break
            time.sleep(10)  # sleep for 10 seconds before the next iteration
            print("-"*150)
        print("="*150)
        break
