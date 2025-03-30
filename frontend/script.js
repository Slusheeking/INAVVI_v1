const API_BASE_URL = 'http://localhost:8001'; // Assuming the FastAPI backend runs on port 8001

const statusContent = document.getElementById('status-content');
const notificationsList = document.getElementById('notifications-list');

function formatTimestamp(unixTimestamp) {
    if (!unixTimestamp) return 'N/A';
    const date = new Date(unixTimestamp * 1000);
    return date.toLocaleString(); // Adjust format as needed
}

async function fetchStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/status`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        renderStatus(data);
    } catch (error) {
        console.error('Error fetching status:', error);
        statusContent.innerHTML = '<span style="color: red;">Error loading status. Is the API running?</span>';
    }
}

function renderStatus(status) {
    const isRunning = status.running === true;
    const statusClass = isRunning ? 'status-running' : 'status-stopped';
    const statusText = isRunning ? 'RUNNING' : 'STOPPED';

    let detailsHtml = '';
    if (status.startup_time) {
        detailsHtml += `<span>Started: ${formatTimestamp(status.startup_time)}</span>`;
    }
    if (status.shutdown_time) {
        detailsHtml += `<span>Stopped: ${formatTimestamp(status.shutdown_time)}</span>`;
    }
     if (status.shutdown_reason) {
        detailsHtml += `<span>Stop Reason: ${status.shutdown_reason}</span>`;
    }
    if (status.last_update) {
        detailsHtml += `<span>Last Update: ${formatTimestamp(status.last_update)}</span>`;
    }
     if (status.last_message) {
        detailsHtml += `<span>Last Message: ${status.last_message}</span>`;
    }
     if (status.last_error) {
        detailsHtml += `<span style="color: red;">Last Error: ${status.last_error}</span>`;
    }


    statusContent.innerHTML = `
        <span class="${statusClass}">${statusText}</span>
        ${detailsHtml}
    `;
}

async function fetchNotifications() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/notifications?limit=50`); // Fetch more notifications
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        renderNotifications(data);
    } catch (error) {
        console.error('Error fetching notifications:', error);
        const li = document.createElement('li');
        li.style.color = 'red';
        li.textContent = 'Error loading notifications.';
        notificationsList.innerHTML = ''; // Clear loading message
        notificationsList.appendChild(li);
    }
}

function renderNotifications(notifications) {
    notificationsList.innerHTML = ''; // Clear previous notifications
    if (notifications.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No notifications yet.';
        notificationsList.appendChild(li);
        return;
    }

    notifications.forEach(n => {
        const li = document.createElement('li');
        const levelClass = `level-${n.level || 'info'}`;
        li.classList.add(levelClass);

        let detailsHtml = '';
        if (n.details && Object.keys(n.details).length > 0) {
            // Basic formatting for details object
            detailsHtml = `<pre class="details">${JSON.stringify(n.details, null, 2)}</pre>`;
        }

        li.innerHTML = `
            <strong>[${n.type || 'General'}]</strong> ${n.message || 'No message'}
            <span class="timestamp">${formatTimestamp(n.timestamp)}</span>
            ${detailsHtml}
        `;
        notificationsList.appendChild(li);
    });
}

// Initial fetch
fetchStatus();
fetchNotifications();

// Refresh data every 5 seconds
setInterval(() => {
    fetchStatus();
    fetchNotifications();
}, 5000);