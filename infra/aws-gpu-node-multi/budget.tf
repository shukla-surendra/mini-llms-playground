# A forgotten pair of g6.xlarge is ~$38/day (2x the single-GPU sibling module's
# ~$19/day estimate) — the backstop for the case the idle watchdog cannot cover
# (disabled by default here, see variables.tf) or a run genuinely runs long.
#
# AWS Budgets emails subscribers directly, so there is no SNS topic to create and
# no subscription-confirmation click to forget.

resource "aws_budgets_budget" "monthly" {
  count = var.monthly_budget_usd > 0 && var.budget_alert_email != "" ? 1 : 0

  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Actual spend crossing 80% — "you are well into it".
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_alert_email]
  }

  # Forecast crossing 100% — the useful one: it fires days before the damage,
  # while an instance you forgot is still only halfway through the month.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_alert_email]
  }
}
