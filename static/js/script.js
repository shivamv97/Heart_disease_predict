/**
 * Heart Disease Detection System - Client-side JavaScript
 * Handles form submission, loading states, and validation
 */

document.addEventListener("DOMContentLoaded", function () {
  // Get the form element
  const patientForm = document.getElementById("patient-form");

  // If the form exists on the current page
  if (patientForm) {
    // Form submission handler
    patientForm.addEventListener("submit", function (event) {
      // Don't prevent default submission - we're using standard form submission
      // But show loading state

      // Validate form before submission
      if (!validateForm()) {
        event.preventDefault();
        return false;
      }

      // Show loading spinner
      const predictBtn = document.getElementById("predict-btn");
      const btnText = document.getElementById("btn-text");
      const loadingSpinner = document.getElementById("loading-spinner");

      btnText.textContent = "Analyzing...";
      loadingSpinner.classList.remove("d-none");
      predictBtn.disabled = true;

      // Continue with form submission
      return true;
    });

    // Initialize tooltips if Bootstrap is loaded
    if (typeof bootstrap !== "undefined") {
      const tooltipTriggerList = [].slice.call(
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
      );
      tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
      });
    }

    // Initialize form field validation listeners
    initializeValidation();
  }

  // Add functionality for resetting form
  const resetBtn = document.getElementById("reset-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      patientForm.reset();
    });
  }

  // Add AJAX prediction functionality (optional enhancement)
  initializeAjaxPrediction();
});

/**
 * Validates form fields before submission
 * @returns {boolean} - Whether the form is valid
 */
function validateForm() {
  // Get all required inputs
  const requiredInputs = document.querySelectorAll(
    "input[required], select[required]"
  );
  let isValid = true;

  // Check each required field
  requiredInputs.forEach((input) => {
    if (!input.value) {
      isValid = false;
      input.classList.add("is-invalid");

      // Add error message if not exists
      const errorDiv = input.nextElementSibling;
      if (!errorDiv || !errorDiv.classList.contains("invalid-feedback")) {
        const div = document.createElement("div");
        div.className = "invalid-feedback";
        div.textContent = "This field is required";
        input.parentNode.insertBefore(div, input.nextSibling);
      }
    } else {
      input.classList.remove("is-invalid");
      // Remove error message if exists
      const errorDiv = input.nextElementSibling;
      if (errorDiv && errorDiv.classList.contains("invalid-feedback")) {
        errorDiv.remove();
      }
    }
  });

  // Validate numeric ranges
  const ageInput = document.getElementById("age");
  if (
    ageInput &&
    (parseInt(ageInput.value) < 20 || parseInt(ageInput.value) > 100)
  ) {
    ageInput.classList.add("is-invalid");
    addErrorMessage(ageInput, "Age must be between 20 and 100");
    isValid = false;
  }

  const bpInput = document.getElementById("trestbps");
  if (
    bpInput &&
    (parseInt(bpInput.value) < 90 || parseInt(bpInput.value) > 200)
  ) {
    bpInput.classList.add("is-invalid");
    addErrorMessage(bpInput, "Blood pressure must be between 90 and 200 mm Hg");
    isValid = false;
  }

  const cholInput = document.getElementById("chol");
  if (
    cholInput &&
    (parseInt(cholInput.value) < 120 || parseInt(cholInput.value) > 400)
  ) {
    cholInput.classList.add("is-invalid");
    addErrorMessage(cholInput, "Cholesterol must be between 120 and 400 mg/dl");
    isValid = false;
  }

  return isValid;
}

/**
 * Add error message after an input field
 * @param {HTMLElement} input - The input element
 * @param {string} message - Error message to display
 */
function addErrorMessage(input, message) {
  // Remove existing error message if any
  const nextEl = input.nextElementSibling;
  if (nextEl && nextEl.classList.contains("invalid-feedback")) {
    nextEl.remove();
  }

  // Add new error message
  const div = document.createElement("div");
  div.className = "invalid-feedback";
  div.textContent = message;
  input.parentNode.insertBefore(div, input.nextSibling);
}

/**
 * Initialize validation event listeners
 */
function initializeValidation() {
  const inputs = document.querySelectorAll("input, select");

  inputs.forEach((input) => {
    input.addEventListener("change", function () {
      if (this.value) {
        this.classList.remove("is-invalid");
        // Remove error message if exists
        const errorDiv = this.nextElementSibling;
        if (errorDiv && errorDiv.classList.contains("invalid-feedback")) {
          errorDiv.remove();
        }
      }
    });
  });
}

/**
 * Initialize AJAX prediction functionality (optional enhancement)
 * This function will enable AJAX form submission if the API endpoint exists
 */
function initializeAjaxPrediction() {
  // This is an optional enhancement that could be implemented
  // to show predictions without full page reloads
  // For now, we'll use the standard form submission
  // but this function could be expanded to use fetch API
  // to submit the form data via AJAX
}
