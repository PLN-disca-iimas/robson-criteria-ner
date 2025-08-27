from utils import RobsonCriteriaCRFNERModel


if __name__ == "__main__":
    base_path = "./"
    iterations = 100
    log_filename = f"snt_crf_{iterations}.log"
    dataset_path = "data/anonymized_gold_dataset_3_sentences"

    rob_crf_ner = RobsonCriteriaCRFNERModel(
        base_path=base_path,
        iterations=iterations,
        model_output_name=f"crf_{iterations}_snts",
    )

    rob_crf_ner.define_logger(log_filename)
    rob_crf_ner.load_dataset(dataset_path)
    rob_crf_ner.main()
