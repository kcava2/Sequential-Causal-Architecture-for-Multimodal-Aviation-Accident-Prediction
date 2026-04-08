import pandas as pd
from train import trainPipeline

def testPipeline() -> tuple[tuple, tuple]:

    # Obtain Training Results
    model_1, model_2, model_3, test_set = trainPipeline()

    # Testing Set
    envrCond_test = test_set[["Light Conditions","Basic Meteorological Conditions","Wind Conditions (kt)","Temperature (C)"]]
    input_1_test = test_set[["Employment Change vs Prior Period (%)","Light Conditions","Basic Meteorological Conditions","Wind Conditions (kt)","Temperature (C)"]]
    persCond_test = test_set["Personnel Conditions"]

    y_A_test = test_set["Supervisory Conditions"]
    y_B_test = test_set["Operator Conditions"]
    y_C_test = test_set["Unsafe Conditions"]


    # Test Input to Model 1
    A_pred_test = model_1.predict(input_1_test)

    # Define Model 2 Inputs
    input_2_test = pd.concat((envrCond_test,persCond_test),axis=1)
    input_2_test['Supervisory Conditions'] = A_pred_test

    # Test Input to Model 2
    B_pred_test = model_2.predict(input_2_test)

    # Define Model 3 Inputs
    input_3_test = pd.DataFrame({
        "Supervisory Conditions" : A_pred_test,
        "Operator Conditions" : B_pred_test
    })

    # Test Input to Model 3
    C_pred_test = model_3.predict(input_3_test)

    # Package Results
    y_test = (y_A_test, y_B_test, y_C_test)
    model_test = (A_pred_test, B_pred_test, C_pred_test)

    return model_test, y_test