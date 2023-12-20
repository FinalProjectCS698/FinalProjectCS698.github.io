// script.js

// Function to show only the selected section
function showResults() {
    var selectedOption = $("#resultsDropdown").val();
    $(".main-content").addClass("hidden");
    $("#" + selectedOption).removeClass("hidden");
}

// Event listener for dropdown change
$(document).ready(function() {
    $("#resultsDropdown").on("change", showResults);
});
