function requiredCheck() {
    var count = 0;
    var PassengerId = $("#PassengerId").val();
    var Name = $("#Name").val();
    var Age = $("#Age").val();
    var SibSp = $("#SibSp").val();
    var Parch = $("#Parch").val();
    var Ticket = $("#Ticket").val();
    var Fare = $("#Fare").val();
    var Cabin = $("#Cabin").val();
    checkNullField(PassengerId, 1);
    checkNullField(Name, 2);
    checkNullField(Age, 3);
    checkNullField(SibSp, 4);
    checkNullField(Parch, 5);
    checkNullField(Ticket, 6);
    checkNullField(Fare, 7);
    checkNullField(Cabin, 8);
    if (count > 0) {
        alert("Fill all details");
        return false;

    } else {
        return $('#fdata').attr('action', '/decisionTree-result');
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
    return (val == 1) ? $("#PassengerId").addClass("errorMessage")
        : (val == 2) ? $("#Name").addClass("errorMessage")
            : (val == 3) ? $("#Age").addClass("errorMessage")
                : (val == 4) ? $("#SibSp").addClass("errorMessage")
                    : (val == 5) ? $("#Parch").addClass("errorMessage")
                        : (val == 6) ? $("#Ticket").addClass("errorMessage")
                            : (val == 7) ? $("#Fare").addClass("errorMessage")
                                : (val == 8) ? $("#Cabin").addClass("errorMessage")
                                    : "";
}
function addCssSucess(val) {
    return (val == 1) ? $("#PassengerId").removeClass("successMessage")
        : (val == 2) ? $("#Name").removeClass("successMessage")
            : (val == 3) ? $("#Age").removeClass("successMessage")
                : (val == 4) ? $("#SibSp").removeClass("successMessage")
                    : (val == 5) ? $("#Parch").removeClass("successMessage")
                        : (val == 6) ? $("#Ticket").removeClass("successMessage")
                            : (val == 7) ? $("#Fare").removeClass("successMessage")
                                : (val == 8) ? $("#Cabin").removeClass("successMessage")
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

