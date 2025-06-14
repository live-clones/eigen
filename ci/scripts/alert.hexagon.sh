#!/bin/bash

# Alerting and Notification System for Hexagon CI Pipeline
# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2023, The Eigen Authors
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== Hexagon CI Alerting System ===${NC}"

# Configuration
ALERT_LOG="${EIGEN_CI_BUILDDIR:-.build}/alerts.log"
ALERT_STATE="${EIGEN_CI_BUILDDIR:-.build}/alert-state.json"
NOTIFICATION_HISTORY="${EIGEN_CI_BUILDDIR:-.build}/notification-history.log"

# Alert settings
ALERT_ENABLED=${EIGEN_ALERT_ENABLED:-true}
ALERT_LEVEL=${EIGEN_ALERT_LEVEL:-warning}  # debug, info, warning, error, critical
ALERT_COOLDOWN=${EIGEN_ALERT_COOLDOWN:-300}  # seconds
ALERT_MAX_RATE=${EIGEN_ALERT_MAX_RATE:-10}   # alerts per hour

# Notification channels
SLACK_WEBHOOK_URL=${EIGEN_SLACK_WEBHOOK_URL:-}
TEAMS_WEBHOOK_URL=${EIGEN_TEAMS_WEBHOOK_URL:-}
EMAIL_RECIPIENTS=${EIGEN_EMAIL_RECIPIENTS:-}
DISCORD_WEBHOOK_URL=${EIGEN_DISCORD_WEBHOOK_URL:-}

# Create alert directories
mkdir -p "$(dirname "${ALERT_LOG}")"

# Initialize alert state
if [ ! -f "$ALERT_STATE" ]; then
    cat > "$ALERT_STATE" << EOF
{
  "last_alert_time": 0,
  "alert_count": 0,
  "rate_limit_resets": 0,
  "alerts_this_hour": 0,
  "hour_start": $(date +%s),
  "active_alerts": {},
  "suppressed_alerts": []
}
EOF
fi

# Alert levels
declare -A ALERT_LEVELS
ALERT_LEVELS[debug]=0
ALERT_LEVELS[info]=1
ALERT_LEVELS[warning]=2
ALERT_LEVELS[error]=3
ALERT_LEVELS[critical]=4

# Get current alert level threshold
ALERT_THRESHOLD=${ALERT_LEVELS[$ALERT_LEVEL]}

# Utility functions
get_timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_alert() {
    local level="$1"
    local title="$2"
    local message="$3"
    local context="$4"
    local timestamp=$(get_timestamp)
    
    cat >> "$ALERT_LOG" << EOF
[$timestamp] LEVEL: $level
[$timestamp] TITLE: $title
[$timestamp] MESSAGE: $message
[$timestamp] CONTEXT: $context
[$timestamp] JOB: ${CI_JOB_NAME:-local}
[$timestamp] COMMIT: ${CI_COMMIT_SHA:-unknown}
[$timestamp] PIPELINE: ${CI_PIPELINE_SOURCE:-local}
[$timestamp] ---

EOF
}

# Rate limiting
check_rate_limit() {
    local current_time=$(date +%s)
    local hour_start=$(jq -r '.hour_start' "$ALERT_STATE" 2>/dev/null || echo "$current_time")
    local alerts_this_hour=$(jq -r '.alerts_this_hour' "$ALERT_STATE" 2>/dev/null || echo "0")
    
    # Reset hourly counter if needed
    if [ $((current_time - hour_start)) -gt 3600 ]; then
        jq --argjson current_time "$current_time" \
           '.hour_start = $current_time | .alerts_this_hour = 0' \
           "$ALERT_STATE" > "${ALERT_STATE}.tmp" && mv "${ALERT_STATE}.tmp" "$ALERT_STATE"
        alerts_this_hour=0
    fi
    
    # Check rate limit
    if [ "$alerts_this_hour" -ge "$ALERT_MAX_RATE" ]; then
        echo -e "${YELLOW}⚠ Alert rate limit exceeded (${alerts_this_hour}/${ALERT_MAX_RATE}), suppressing alert${NC}"
        return 1
    fi
    
    return 0
}

# Cooldown checking
check_cooldown() {
    local alert_key="$1"
    local current_time=$(date +%s)
    local last_alert_time=$(jq -r --arg key "$alert_key" '.active_alerts[$key] // 0' "$ALERT_STATE" 2>/dev/null || echo "0")
    
    if [ $((current_time - last_alert_time)) -lt "$ALERT_COOLDOWN" ]; then
        echo -e "${YELLOW}⚠ Alert '$alert_key' is in cooldown, suppressing${NC}"
        return 1
    fi
    
    return 0
}

# Update alert state
update_alert_state() {
    local alert_key="$1"
    local current_time=$(date +%s)
    
    # Update state file
    jq --argjson current_time "$current_time" \
       --arg alert_key "$alert_key" \
       '.last_alert_time = $current_time | 
        .alert_count += 1 | 
        .alerts_this_hour += 1 | 
        .active_alerts[$alert_key] = $current_time' \
       "$ALERT_STATE" > "${ALERT_STATE}.tmp" && mv "${ALERT_STATE}.tmp" "$ALERT_STATE"
}

# Slack notification
send_slack_notification() {
    local level="$1"
    local title="$2"
    local message="$3"
    local context="$4"
    
    if [ -z "$SLACK_WEBHOOK_URL" ]; then
        return 0
    fi
    
    local color="good"
    local emoji=":white_check_mark:"
    
    case "$level" in
        "error"|"critical")
            color="danger"
            emoji=":x:"
            ;;
        "warning")
            color="warning"
            emoji=":warning:"
            ;;
        "info")
            color="good"
            emoji=":information_source:"
            ;;
    esac
    
    local payload=$(cat << EOF
{
  "attachments": [
    {
      "color": "$color",
      "title": "$emoji Hexagon CI Alert: $title",
      "text": "$message",
      "fields": [
        {
          "title": "Level",
          "value": "$level",
          "short": true
        },
        {
          "title": "Job",
          "value": "${CI_JOB_NAME:-local}",
          "short": true
        },
        {
          "title": "Commit",
          "value": "${CI_COMMIT_SHA:-unknown}",
          "short": true
        },
        {
          "title": "Pipeline",
          "value": "${CI_PIPELINE_SOURCE:-local}",
          "short": true
        },
        {
          "title": "Context",
          "value": "$context",
          "short": false
        }
      ],
      "footer": "Hexagon CI",
      "ts": $(date +%s)
    }
  ]
}
EOF
)
    
    if curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL" >/dev/null; then
        echo -e "${GREEN}✓ Slack notification sent${NC}"
        log_notification "slack" "$level" "$title" "success"
    else
        echo -e "${RED}✗ Failed to send Slack notification${NC}"
        log_notification "slack" "$level" "$title" "failed"
    fi
}

# Teams notification
send_teams_notification() {
    local level="$1"
    local title="$2"
    local message="$3"
    local context="$4"
    
    if [ -z "$TEAMS_WEBHOOK_URL" ]; then
        return 0
    fi
    
    local theme_color="00FF00"
    
    case "$level" in
        "error"|"critical")
            theme_color="FF0000"
            ;;
        "warning")
            theme_color="FFA500"
            ;;
        "info")
            theme_color="0078D4"
            ;;
    esac
    
    local payload=$(cat << EOF
{
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "themeColor": "$theme_color",
  "summary": "Hexagon CI Alert: $title",
  "sections": [
    {
      "activityTitle": "Hexagon CI Alert",
      "activitySubtitle": "$title",
      "activityImage": "https://eigen.tuxfamily.org/dox/eigen_logo.png",
      "facts": [
        {
          "name": "Level",
          "value": "$level"
        },
        {
          "name": "Job",
          "value": "${CI_JOB_NAME:-local}"
        },
        {
          "name": "Commit",
          "value": "${CI_COMMIT_SHA:-unknown}"
        },
        {
          "name": "Pipeline",
          "value": "${CI_PIPELINE_SOURCE:-local}"
        },
        {
          "name": "Message",
          "value": "$message"
        },
        {
          "name": "Context",
          "value": "$context"
        }
      ],
      "markdown": true
    }
  ]
}
EOF
)
    
    if curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$TEAMS_WEBHOOK_URL" >/dev/null; then
        echo -e "${GREEN}✓ Teams notification sent${NC}"
        log_notification "teams" "$level" "$title" "success"
    else
        echo -e "${RED}✗ Failed to send Teams notification${NC}"
        log_notification "teams" "$level" "$title" "failed"
    fi
}

# Discord notification
send_discord_notification() {
    local level="$1"
    local title="$2"
    local message="$3"
    local context="$4"
    
    if [ -z "$DISCORD_WEBHOOK_URL" ]; then
        return 0
    fi
    
    local color=65280  # Green
    local emoji=""
    
    case "$level" in
        "error"|"critical")
            color=16711680  # Red
            emoji="🚨"
            ;;
        "warning")
            color=16753920  # Orange
            emoji="⚠️"
            ;;
        "info")
            color=3447003   # Blue
            emoji="ℹ️"
            ;;
    esac
    
    local payload=$(cat << EOF
{
  "embeds": [
    {
      "title": "$emoji Hexagon CI Alert: $title",
      "description": "$message",
      "color": $color,
      "fields": [
        {
          "name": "Level",
          "value": "$level",
          "inline": true
        },
        {
          "name": "Job",
          "value": "${CI_JOB_NAME:-local}",
          "inline": true
        },
        {
          "name": "Commit",
          "value": "${CI_COMMIT_SHA:-unknown}",
          "inline": true
        },
        {
          "name": "Context",
          "value": "$context",
          "inline": false
        }
      ],
      "footer": {
        "text": "Hexagon CI"
      },
      "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    }
  ]
}
EOF
)
    
    if curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$DISCORD_WEBHOOK_URL" >/dev/null; then
        echo -e "${GREEN}✓ Discord notification sent${NC}"
        log_notification "discord" "$level" "$title" "success"
    else
        echo -e "${RED}✗ Failed to send Discord notification${NC}"
        log_notification "discord" "$level" "$title" "failed"
    fi
}

# Email notification
send_email_notification() {
    local level="$1"
    local title="$2"
    local message="$3"
    local context="$4"
    
    if [ -z "$EMAIL_RECIPIENTS" ] || ! command -v mail >/dev/null 2>&1; then
        return 0
    fi
    
    local subject="[Hexagon CI] $level: $title"
    local body=$(cat << EOF
Hexagon CI Alert

Level: $level
Title: $title
Message: $message
Context: $context

Job Details:
- Job Name: ${CI_JOB_NAME:-local}
- Commit SHA: ${CI_COMMIT_SHA:-unknown}
- Pipeline Source: ${CI_PIPELINE_SOURCE:-local}
- Timestamp: $(get_timestamp)

This is an automated message from the Hexagon CI pipeline.
EOF
)
    
    if echo "$body" | mail -s "$subject" "$EMAIL_RECIPIENTS"; then
        echo -e "${GREEN}✓ Email notification sent${NC}"
        log_notification "email" "$level" "$title" "success"
    else
        echo -e "${RED}✗ Failed to send email notification${NC}"
        log_notification "email" "$level" "$title" "failed"
    fi
}

# Log notification attempts
log_notification() {
    local channel="$1"
    local level="$2"
    local title="$3"
    local status="$4"
    local timestamp=$(get_timestamp)
    
    echo "[$timestamp] CHANNEL: $channel, LEVEL: $level, TITLE: $title, STATUS: $status" >> "$NOTIFICATION_HISTORY"
}

# Main alert function
send_alert() {
    local level="$1"
    local title="$2"
    local message="$3"
    local context="${4:-}"
    
    # Check if alerting is enabled
    if [ "$ALERT_ENABLED" != "true" ]; then
        echo -e "${YELLOW}⚠ Alerting is disabled${NC}"
        return 0
    fi
    
    # Check alert level threshold
    local level_value=${ALERT_LEVELS[$level]:-0}
    if [ "$level_value" -lt "$ALERT_THRESHOLD" ]; then
        echo -e "${YELLOW}⚠ Alert level '$level' below threshold '$ALERT_LEVEL'${NC}"
        return 0
    fi
    
    # Create alert key for deduplication
    local alert_key="${level}-${title}"
    
    # Check rate limiting
    if ! check_rate_limit; then
        return 0
    fi
    
    # Check cooldown
    if ! check_cooldown "$alert_key"; then
        return 0
    fi
    
    echo -e "${BLUE}📢 Sending alert: $level - $title${NC}"
    
    # Log the alert
    log_alert "$level" "$title" "$message" "$context"
    
    # Send notifications to all configured channels
    send_slack_notification "$level" "$title" "$message" "$context" &
    send_teams_notification "$level" "$title" "$message" "$context" &
    send_discord_notification "$level" "$title" "$message" "$context" &
    send_email_notification "$level" "$title" "$message" "$context" &
    
    # Wait for all notifications to complete
    wait
    
    # Update alert state
    update_alert_state "$alert_key"
    
    echo -e "${GREEN}✓ Alert processing completed${NC}"
}

# Predefined alert types
alert_build_failure() {
    local build_type="$1"
    local error_details="$2"
    
    send_alert "error" "Build Failure" \
        "Hexagon build failed for configuration: $build_type" \
        "Error details: $error_details"
}

alert_test_failure() {
    local test_name="$1"
    local failure_count="$2"
    
    send_alert "warning" "Test Failures" \
        "Hexagon tests failed: $failure_count failures in $test_name" \
        "Please check test logs for detailed failure information"
}

alert_toolchain_issue() {
    local issue_type="$1"
    local details="$2"
    
    send_alert "error" "Toolchain Issue" \
        "Hexagon toolchain problem: $issue_type" \
        "Details: $details"
}

alert_performance_degradation() {
    local metric="$1"
    local current_value="$2"
    local baseline="$3"
    
    send_alert "warning" "Performance Degradation" \
        "Performance regression detected in $metric: $current_value (baseline: $baseline)" \
        "This may indicate a performance regression that needs investigation"
}

alert_resource_exhaustion() {
    local resource_type="$1"
    local usage_percent="$2"
    
    send_alert "critical" "Resource Exhaustion" \
        "High $resource_type usage detected: ${usage_percent}%" \
        "The system may be running out of resources, which could cause build failures"
}

alert_pipeline_success() {
    local build_time="$1"
    local test_count="$2"
    
    send_alert "info" "Pipeline Success" \
        "Hexagon CI pipeline completed successfully in ${build_time}s with $test_count tests passed" \
        "All build variants and tests completed without issues"
}

# Alert configuration management
configure_alerts() {
    local config_file="$1"
    
    if [ -f "$config_file" ]; then
        echo -e "${BLUE}Loading alert configuration from: $config_file${NC}"
        
        # Source configuration file
        source "$config_file"
        
        echo -e "${GREEN}✓ Alert configuration loaded${NC}"
        echo "  Alert Level: $ALERT_LEVEL"
        echo "  Alert Cooldown: $ALERT_COOLDOWN seconds"
        echo "  Max Rate: $ALERT_MAX_RATE alerts/hour"
        echo "  Slack: $([ -n "$SLACK_WEBHOOK_URL" ] && echo "configured" || echo "not configured")"
        echo "  Teams: $([ -n "$TEAMS_WEBHOOK_URL" ] && echo "configured" || echo "not configured")"
        echo "  Discord: $([ -n "$DISCORD_WEBHOOK_URL" ] && echo "configured" || echo "not configured")"
        echo "  Email: $([ -n "$EMAIL_RECIPIENTS" ] && echo "configured" || echo "not configured")"
    else
        echo -e "${YELLOW}⚠ Alert configuration file not found: $config_file${NC}"
    fi
}

# Alert testing
test_alerts() {
    echo -e "${BLUE}=== Testing Alert System ===${NC}"
    
    send_alert "info" "Test Alert" "This is a test alert to verify the notification system" "Alert system test"
    
    echo -e "${GREEN}✓ Test alert sent${NC}"
}

# Alert statistics
show_alert_statistics() {
    echo -e "${CYAN}=== Alert Statistics ===${NC}"
    
    if [ -f "$ALERT_STATE" ]; then
        local total_alerts=$(jq -r '.alert_count' "$ALERT_STATE")
        local alerts_this_hour=$(jq -r '.alerts_this_hour' "$ALERT_STATE")
        local last_alert_time=$(jq -r '.last_alert_time' "$ALERT_STATE")
        
        echo "Total Alerts: $total_alerts"
        echo "Alerts This Hour: $alerts_this_hour"
        echo "Last Alert: $(date -d "@$last_alert_time" 2>/dev/null || echo "Never")"
    fi
    
    if [ -f "$NOTIFICATION_HISTORY" ]; then
        echo ""
        echo "Recent Notifications:"
        tail -10 "$NOTIFICATION_HISTORY" | while read -r line; do
            echo "  $line"
        done
    fi
}

# Cleanup old alerts
cleanup_alerts() {
    local days_to_keep="${1:-7}"
    
    echo -e "${BLUE}Cleaning up alerts older than $days_to_keep days...${NC}"
    
    # Clean up log files
    if [ -f "$ALERT_LOG" ]; then
        find "$(dirname "$ALERT_LOG")" -name "*.log" -mtime +$days_to_keep -delete 2>/dev/null || true
    fi
    
    if [ -f "$NOTIFICATION_HISTORY" ]; then
        find "$(dirname "$NOTIFICATION_HISTORY")" -name "notification-history.log" -mtime +$days_to_keep -delete 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Alert cleanup completed${NC}"
}

# Main function
main() {
    local command="$1"
    shift || true
    
    case "$command" in
        "send"|"alert")
            send_alert "$@"
            ;;
        "build_failure")
            alert_build_failure "$@"
            ;;
        "test_failure")
            alert_test_failure "$@"
            ;;
        "toolchain_issue")
            alert_toolchain_issue "$@"
            ;;
        "performance_degradation")
            alert_performance_degradation "$@"
            ;;
        "resource_exhaustion")
            alert_resource_exhaustion "$@"
            ;;
        "pipeline_success")
            alert_pipeline_success "$@"
            ;;
        "configure")
            configure_alerts "$@"
            ;;
        "test")
            test_alerts
            ;;
        "stats"|"statistics")
            show_alert_statistics
            ;;
        "cleanup")
            cleanup_alerts "$@"
            ;;
        *)
            echo -e "${RED}Unknown command: $command${NC}"
            echo "Available commands:"
            echo "  send <level> <title> <message> [context]"
            echo "  build_failure <build_type> <error_details>"
            echo "  test_failure <test_name> <failure_count>"
            echo "  toolchain_issue <issue_type> <details>"
            echo "  performance_degradation <metric> <current> <baseline>"
            echo "  resource_exhaustion <resource_type> <usage_percent>"
            echo "  pipeline_success <build_time> <test_count>"
            echo "  configure <config_file>"
            echo "  test"
            echo "  stats"
            echo "  cleanup [days_to_keep]"
            exit 1
            ;;
    esac
}

# If script is called directly, run the main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi 