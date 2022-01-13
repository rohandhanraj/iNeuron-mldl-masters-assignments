function requiredCheck() {
    var count = 0;
    var CRIM = $("#CRIM").val();
    var ZN = $("#ZN").val();
    var INDUS = $("#INDUS").val();
    var NOX = $("#NOX").val();
    var RM = $("#RM").val();
    var AGE = $("#AGE").val();
    var DIS = $("#DIS").val();
    var RAD = $("#RAD").val();
    var TAX = $("#TAX").val();
    var PTRATIO = $("#PTRATIO").val();
    var B = $("#B").val();
    var LSTAT = $("#LSTAT").val();
    checkNullField(CRIM, 1);
    checkNullField(ZN, 2);
    checkNullField(INDUS, 3);
    checkNullField(NOX, 4);
    checkNullField(RM, 5);
    checkNullField(AGE, 6);
    checkNullField(DIS, 7);
    checkNullField(RAD, 8);
    checkNullField(TAX, 9);
    checkNullField(PTRATIO, 10);
    checkNullField(B, 11);
    checkNullField(LSTAT, 12);
    if (count > 0) {
        alert("Fill all details");
        return false;

    } else {
        return $('#fdata').attr('action', '/randomForest-result');
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
    return (val == 1) ? $("#CRIM").addClass("errorMessage")
        : (val == 2) ? $("#ZN").addClass("errorMessage")
            : (val == 3) ? $("#INDUS").addClass("errorMessage")
                : (val == 4) ? $("#NOX").addClass("errorMessage")
                    : (val == 5) ? $("#RM").addClass("errorMessage")
                        : (val == 6) ? $("#AGE").addClass("errorMessage")
                            : (val == 7) ? $("#DIS").addClass("errorMessage")
                                : (val == 8) ? $("#RAD").addClass("errorMessage")
                                    : (val == 9) ? $("#TAX").addClass("errorMessage")
                                        : (val == 10) ? $("#PTRATIO").addClass("errorMessage")
                                            : (val == 11) ? $("#B").addClass("errorMessage")
                                                : (val == 12) ? $("#LSTAT").addClass("errorMessage")
                                                    : "";
}
function addCssSucess(val) {
    return (val == 1) ? $("#CRIM").removeClass("successMessage")
        : (val == 2) ? $("#ZN").removeClass("successMessage")
            : (val == 3) ? $("#INDUS").removeClass("successMessage")
                : (val == 4) ? $("#NOX").removeClass("successMessage")
                    : (val == 5) ? $("#RM").removeClass("successMessage")
                        : (val == 6) ? $("#AGE").removeClass("successMessage")
                            : (val == 7) ? $("#DIS").removeClass("successMessage")
                                : (val == 8) ? $("#RAD").removeClass("successMessage")
                                    : (val == 9) ? $("#TAX").removeClass("successMessage")
                                        : (val == 10) ? $("#PTRATIO").removeClass("successMessage")
                                            : (val == 11) ? $("#B").removeClass("successMessage")
                                                : (val == 12) ? $("#LSTAT").removeClass("successMessage")
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
