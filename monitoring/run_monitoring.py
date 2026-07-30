import logging

from monitoring.reader import (
    read_features,
    read_predictions
)

from monitoring.metrics import (
    calculate_metrics
)

from monitoring.writer import (
    write_metrics
)


logging.basicConfig(
    level=logging.INFO
)



def main():

    logging.info(
        "Starting monitoring job"
    )


    features = read_features()


    predictions = read_predictions()


    metrics = calculate_metrics(
        features,
        predictions
    )


    output = write_metrics(
        metrics
    )


    logging.info(
        f"Monitoring complete: {output}"
    )



if __name__ == "__main__":

    main()