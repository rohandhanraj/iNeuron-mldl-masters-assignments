function requiredCheck() {
    var count = 0;
    var age = $("#age").val();
    var workclass = $("#workclass").val();
    var fnlwgt = $("#fnlwgt").val();
    var education = $("#education").val();
    var education_num = $("#education_num").val();
    var marital_status = $("#marital_status").val();
    var occupation = $("#occupation").val();
    var relationship = $("#relationship").val();
    var race = $("#race").val();
    var sex = $("#sex").val();
    var capital_gain = $("#capital_gain").val();
    var capital_loss = $("#capital_loss").val();
    var hours_per_week = $("#hours_per_week").val();
    var native_country = $("#native_country").val();
    checkNullField(age, 1);
    checkNullField(workclass, 2);
    checkNullField(fnlwgt, 3);
    checkNullField(education, 4);
    checkNullField(education_num, 5);
    checkNullField(marital_status, 6);
    checkNullField(occupation, 7);
    checkNullField(relationship, 8);
    checkNullField(race, 9);
    checkNullField(sex, 10);
    checkNullField(capital_gain, 11);
    checkNullField(capital_loss, 12);
    checkNullField(hours_per_week, 13);
    checkNullField(native_country, 14);
    if (count > 0) {
        alert("Fill all details");
        return false;

    } else {
        return $('#fdata').attr('action', '/xgBoost-result');
    }

    function checkNullField(id, val) {

        if (id == "" || id == null) {
            count++;
            addCssError(val);
        } else {
            addCssSucess(val);
        }
    }
}

function addCssError(val) {
    return (val == 1) ? $("#age").addClass("errorMessage")
        : (val == 2) ? $("#workclass").addClass("errorMessage")
            : (val == 3) ? $("#fnlwgt").addClass("errorMessage")
                : (val == 4) ? $("#education").addClass("errorMessage")
                    : (val == 5) ? $("#education_num").addClass("errorMessage")
                        : (val == 6) ? $("#marital_status").addClass("errorMessage")
                            : (val == 7) ? $("#occupation").addClass("errorMessage")
                                : (val == 8) ? $("#relationship").addClass("errorMessage")
                                    : (val == 9) ? $("#race").addClass("errorMessage")
                                        : (val == 10) ? $("#sex").addClass("errorMessage")
                                            : (val == 11) ? $("#capital_gain").addClass("errorMessage")
                                                : (val == 12) ? $("#capital_loss").addClass("errorMessage")
                                                    : (val == 13) ? $("#hours_per_week").addClass("errorMessage")
                                                        : (val == 14) ? $("#native_country").addClass("errorMessage")
                                                            : "";
}
function addCssSucess(val) {
    return (val == 1) ? $("#age").removeClass("successMessage")
        : (val == 2) ? $("#workclass").removeClass("successMessage")
            : (val == 3) ? $("#fnlwgt").removeClass("successMessage")
                : (val == 4) ? $("#education").removeClass("successMessage")
                    : (val == 5) ? $("#education_num").removeClass("successMessage")
                        : (val == 6) ? $("#marital_status").removeClass("successMessage")
                            : (val == 7) ? $("#occupation").removeClass("successMessage")
                                : (val == 8) ? $("#relationship").removeClass("successMessage")
                                    : (val == 9) ? $("#race").removeClass("successMessage")
                                        : (val == 10) ? $("#sex").removeClass("successMessage")
                                            : (val == 11) ? $("#capital_gain").removeClass("successMessage")
                                                : (val == 12) ? $("#capital_loss").removeClass("successMessage")
                                                    : (val == 13) ? $("#hours_per_week").removeClass("successMessage")
                                                        : (val == 14) ? $("#native_country").removeClass("successMessage")
                                                            : "";
}

function clear() {
    var elements = document.getElementsByTagName("input");
    for (var i = 0; i < elements.length; i++) {
        if (elements[i].type == "text") {
            elements[i].value = "";
        }
    }
}

