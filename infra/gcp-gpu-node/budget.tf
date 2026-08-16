# A forgotten g2-standard-4 is real, ongoing money. This is the backstop for what the
# idle watchdog cannot cover: an instance genuinely busy doing something you forgot
# you started, or storage cost drifting up on its own. Same purpose as the AWS
# module's aws_budgets_budget — the mechanics differ more than the syntax, though:
#
# - AWS Budgets emails a plain address directly. GCP billing budgets notify every
#   principal with a Billing Account Administrator/User IAM role by default — that
#   happens with NO extra resource below, automatically, the moment a budget exists.
# - A specific extra address (budget_alert_email) needs its own Cloud Monitoring
#   notification channel, created here and attached via all_updates_rule — an
#   AWS Budgets subscriber list has no equivalent extra resource to create.

resource "google_monitoring_notification_channel" "budget_email" {
  count = var.monthly_budget_usd > 0 && var.billing_account_id != "" && var.budget_alert_email != "" ? 1 : 0

  project      = var.project_id
  display_name = "${var.project} budget alerts"
  type         = "email"

  labels = {
    email_address = var.budget_alert_email
  }
}

resource "google_billing_budget" "monthly" {
  count = var.monthly_budget_usd > 0 && var.billing_account_id != "" ? 1 : 0

  billing_account = "billingAccounts/${var.billing_account_id}"
  display_name    = "${var.project}-monthly"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_budget_usd)
    }
  }

  # Actual spend crossing 80% — "you are well into it". Same threshold as the AWS module.
  threshold_rules {
    threshold_percent = 0.8
    spend_basis       = "CURRENT_SPEND"
  }

  # Forecast crossing 100% — the useful one: it fires days before the damage, while
  # an instance you forgot is still only halfway through the month. Same as the AWS
  # module's FORECASTED notification.
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  dynamic "all_updates_rule" {
    for_each = length(google_monitoring_notification_channel.budget_email) > 0 ? [1] : []

    content {
      monitoring_notification_channels = [google_monitoring_notification_channel.budget_email[0].id]
      # false = also keep notifying every Billing Account Admin/User (GCP's default
      # audience) rather than replacing it with only the address above.
      disable_default_iam_recipients = false
    }
  }
}
