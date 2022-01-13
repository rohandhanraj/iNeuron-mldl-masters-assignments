function requiredCheck() {
    var count = 0;
    var rate_marriage = $("#rate_marriage").val();
    var age = $("#age").val();
    var yrs_married = $("#yrs_married").val();
    var children = $("#children").val();
    var religious = $("#religious").val();
    var educ = $("#educ").val();
    var occupation = $("#occupation").val();
    var occupation_husb = $("#occupation_husb").val();
    checkNullField(rate_marriage, 1);
    checkNullField(age, 2);
    checkNullField(yrs_married, 3);
    checkNullField(children, 4);
    checkNullField(religious, 5);
    checkNullField(educ, 6);
    checkNullField(occupation, 7);
    checkNullField(occupation_husb, 8);
    if (count > 0) {
        alert("Fill all details");
        return false;

    } else {
        return $('#fdata').attr('action', '/logisticRegression-result');
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
    return (val == 1) ? $("#rate_marriage").addClass("errorMessage")
        : (val == 2) ? $("#age").addClass("errorMessage")
            : (val == 3) ? $("#yrs_married").addClass("errorMessage")
                : (val == 4) ? $("#children").addClass("errorMessage")
                    : (val == 5) ? $("#religious").addClass("errorMessage")
                        : (val == 6) ? $("#educ").addClass("errorMessage")
                            : (val == 7) ? $("#occupation").addClass("errorMessage")
                                : (val == 8) ? $("#occupation_husb").addClass("errorMessage")
                                    : "";
}
function addCssSucess(val) {
    return (val == 1) ? $("#rate_marriage").removeClass("successMessage")
        : (val == 2) ? $("#age").removeClass("successMessage")
            : (val == 3) ? $("#yrs_married").removeClass("successMessage")
                : (val == 4) ? $("#children").removeClass("successMessage")
                    : (val == 5) ? $("#religious").removeClass("successMessage")
                        : (val == 6) ? $("#educ").removeClass("successMessage")
                            : (val == 7) ? $("#occupation").removeClass("successMessage")
                                : (val == 8) ? $("#occupation_husb").removeClass("successMessage")
                                    : "";
}

function clear() {
    var elements = document.getElementsByTagName("input");
  for (var i=0; i < elements.length; i++) {
        if (elements[i].type == "text") {
          elements[i].value = "";
        }
  }
}
