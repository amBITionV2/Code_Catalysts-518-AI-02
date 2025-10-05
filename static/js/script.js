document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predict-btn');
    const loadingElement = document.getElementById('loading');
    const resultsElement = document.getElementById('results');
    const resultsContainer = document.getElementById('results-container');
    const errorElement = document.getElementById('error');
    const topnSlider = document.getElementById('topn');
    const topnValue = document.getElementById('topn-value');

    // Update the value display for the range input
    topnSlider.addEventListener('input', function() {
        topnValue.textContent = this.value;
    });

    // Handle form submission
    predictBtn.addEventListener('click', function() {
        const rank = document.getElementById('rank').value;
        const category = document.getElementById('category').value;
        const location = document.getElementById('location').value;
        const branch = document.getElementById('branch').value;
        const topn = topnSlider.value;

        // Validate inputs
        if (!rank) {
            showError('Please enter your KCET rank');
            return;
        }

        // Show loading state
        loadingElement.classList.remove('hidden');
        resultsElement.classList.add('hidden');
        errorElement.classList.add('hidden');

        // Make API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                rank: parseInt(rank),
                category: category,
                location: location,
                branch: branch,
                topn: parseInt(topn)
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(responseData => {
            console.log('API Response:', responseData);
            
            // Handle the response data
            if (responseData && responseData.success !== false) {
                // Check for nested data array or use the response directly
                const colleges = Array.isArray(responseData.data) ? responseData.data : 
                               (Array.isArray(responseData) ? responseData : []);
                
                console.log('Colleges data to display:', colleges);
                
                if (colleges.length > 0) {
                    displayResults(colleges);
                } else {
                    throw new Error('No colleges found matching your criteria');
                }
            } else {
                throw new Error(responseData.error || 'Failed to get recommendations');
            }
        })
        .catch(error => {
            showError(error.message || 'An error occurred. Please try again.');
        })
        .finally(() => {
        });
    });

    // Display results
    function displayResults(colleges) {
        console.log('Displaying colleges:', colleges);
        
        try {
            // Clear any previous results and errors
            resultsContainer.innerHTML = '';
            errorElement.classList.add('hidden');
            
            if (!colleges || !Array.isArray(colleges) || colleges.length === 0) {
                console.error('No valid colleges data received');
                showError('No colleges found matching your criteria');
                return;
            }
            
            // Sort colleges by admission probability (highest first)
            colleges.sort((a, b) => {
                const probA = a.admission_probability || 0;
                const probB = b.admission_probability || 0;
                return probB - probA; // Sort descending
            });
            
            // Make sure the results container is visible
            resultsElement.classList.remove('hidden');

            // Process each college
            colleges.forEach(college => {
            const collegeCard = document.createElement('div');
            collegeCard.className = 'college-card';

            // Format the college name
            const formattedName = college.college_name
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');

            // Determine chance of admission based on probability
            let chanceClass = 'chance-medium';
            let chanceText = 'Medium Chance';
            const probPercent = Math.round(college.admission_probability * 100);
            
            if (college.admission_probability >= 0.8) {
                chanceClass = 'chance-high';
                chanceText = 'High Chance';
            } else if (college.admission_probability < 0.6) {
                chanceClass = 'chance-low';
                chanceText = 'Low Chance';
            }

            collegeCard.innerHTML = `
                <div class="college-name">${formattedName} <span class="college-code">${college.college_code || ''}</span></div>
                <div class="college-details">
                    <div class="detail-item">
                        <div class="detail-label">Branch</div>
                        <div class="detail-value">${college.branch || 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Category</div>
                        <div class="detail-value">${college.category || 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Last Year Cutoff</div>
                        <div class="detail-value">${college.last_year_cutoff ? 'Rank ' + college.last_year_cutoff.toLocaleString() : 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Admission Probability</div>
                        <div class="detail-value ${chanceClass}">
                            ${chanceText} (${probPercent}%)
                        </div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Location</div>
                        <div class="detail-value">${college.location || 'N/A'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Type</div>
                        <div class="detail-value">${college.institute_type || 'N/A'}</div>
                    </div>
                </div>
            `;

                resultsContainer.appendChild(collegeCard);
            });
        } catch (error) {
            console.error('Error displaying results:', error);
            showError('An error occurred while displaying results. Please try again.');
        } finally {
            resultsElement.classList.remove('hidden');
        }
    }

    // Show error message
    function showError(message) {
        errorElement.classList.remove('hidden');
        document.getElementById('error-message').textContent = message;
    }
});
