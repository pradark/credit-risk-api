import datetime
import boto3

from app.config import (
    S3_BUCKET,
    MIN_PRODUCTION_SAMPLES
)

from app.monitoring.psi import (
    calculate_feature_psi
)

from app.monitoring.s3_utils import (
    read_parquet_from_s3,
    write_parquet_to_s3
)


# AWS client

cloudwatch = boto3.client(
    "cloudwatch",
    region_name="us-east-1"
)


def run_monitoring():

    # Date partition

    today = datetime.date.today().isoformat()


    # S3 locations

    reference_key = (
        "monitoring/reference/"
        "reference_data.parquet"
    )


    production_key = (
        f"features/dt={today}/"
        "data.parquet"
    )


    output_key = (
        f"monitoring/drift/"
        f"psi_{today}.parquet"
    )


    print("Loading reference data...")


    reference = read_parquet_from_s3(
        S3_BUCKET,
        reference_key
    )


    print("Loading production data...")


    production = read_parquet_from_s3(
        S3_BUCKET,
        production_key
    )


    #
    # Production data validation
    #

    if len(production) < MIN_PRODUCTION_SAMPLES:

        raise ValueError(
            f"Not enough production samples for PSI. "
            f"Found {len(production)}, "
            f"need at least {MIN_PRODUCTION_SAMPLES}."
        )


    print("Calculating PSI...")


    psi_results = calculate_feature_psi(
        reference,
        production
    )


    #
    # Monitoring metadata
    #

    psi_results["run_date"] = today
    psi_results["reference_dataset"] = reference_key
    psi_results["production_dataset"] = production_key
    psi_results["sample_size"] = len(production)
    psi_results["model_version"] = "credit-risk-model-v1"


    print()
    print("PSI Results")
    print("-" * 50)

    print(
        psi_results.to_string(
            index=False
        )
    )


    #
    # Write PSI report
    #

    write_parquet_to_s3(
        psi_results,
        S3_BUCKET,
        output_key
    )


    print(
        f"Successfully wrote PSI report to S3: {output_key}"
    )


    #
    # Publish CloudWatch metrics
    #

    for _, row in psi_results.iterrows():

        cloudwatch.put_metric_data(

            Namespace="CreditRiskAPI",

            MetricData=[

                {

                    "MetricName":
                        f"FeaturePSI_{row['feature']}",

                    "Value":
                        float(row["psi"]),

                    "Unit":
                        "None"

                }

            ]

        )


    print(
        "Successfully published PSI metrics to CloudWatch"
    )


    print(
        "Monitoring complete"
    )


    return psi_results



if __name__ == "__main__":

    run_monitoring()