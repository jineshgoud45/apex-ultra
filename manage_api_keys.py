import os
from dotenv import load_dotenv

# List of required API keys and credentials for the bot's revenue streams and integrations
REQUIRED_KEYS = [
    {
        "name": "REVENUE_API_KEY",
        "description": "Main API key for revenue engine and stream management.",
        "where_to_get": "Obtain from your revenue platform or admin dashboard.",
        "example": "REVENUE_API_KEY=your_revenue_api_key_here"
    },
    {
        "name": "ADS_API_KEY",
        "description": "API key for ad network integration (e.g., Google Ads, Facebook Ads).",
        "where_to_get": "Sign up for the ad network and generate an API key in the developer console.",
        "example": "ADS_API_KEY=your_ads_api_key_here"
    },
    {
        "name": "AFFILIATE_API_KEY",
        "description": "API key for affiliate marketing platform integration.",
        "where_to_get": "Register with your affiliate platform (e.g., Amazon Associates, Impact) and generate an API key.",
        "example": "AFFILIATE_API_KEY=your_affiliate_api_key_here"
    },
    {
        "name": "SAAS_BILLING_API_KEY",
        "description": "API key for SaaS billing/payment processor integration (e.g., Stripe, PayPal).",
        "where_to_get": "Create an account with your payment processor and generate an API key in the dashboard.",
        "example": "SAAS_BILLING_API_KEY=your_billing_api_key_here"
    },
    {
        "name": "MARKETPLACE_API_KEY",
        "description": "API key for marketplace or e-commerce platform integration (e.g., Shopify, WooCommerce).",
        "where_to_get": "Register your app with the marketplace platform and generate an API key.",
        "example": "MARKETPLACE_API_KEY=your_marketplace_api_key_here"
    },
    {
        "name": "LICENSING_API_KEY",
        "description": "API key for licensing or IP management platform.",
        "where_to_get": "Sign up with your licensing provider and generate an API key.",
        "example": "LICENSING_API_KEY=your_licensing_api_key_here"
    },
    {
        "name": "CONTENT_API_KEY",
        "description": "API key for content creation or publishing platform integration (e.g., YouTube, Medium, WordPress).",
        "where_to_get": "Register your app with the content platform and generate an API key.",
        "example": "CONTENT_API_KEY=your_content_api_key_here"
    },
    {
        "name": "TRADING_API_KEY",
        "description": "API key for trading or financial data integration (e.g., Alpaca, Binance, Yahoo Finance).",
        "where_to_get": "Create an account with your trading platform and generate an API key.",
        "example": "TRADING_API_KEY=your_trading_api_key_here"
    },
    {
        "name": "ANALYTICS_API_KEY",
        "description": "API key for analytics/monitoring platform (e.g., Prometheus, Grafana Cloud, Google Analytics).",
        "where_to_get": "Register with your analytics provider and generate an API key.",
        "example": "ANALYTICS_API_KEY=your_analytics_api_key_here"
    },
    {
        "name": "EMAIL_API_KEY",
        "description": "API key for transactional email or notification service (e.g., SendGrid, Mailgun, SES).",
        "where_to_get": "Sign up for your email provider and generate an API key.",
        "example": "EMAIL_API_KEY=your_email_api_key_here"
    },
    # Add more as needed for your integrations
]

def print_api_key_checklist():
    print("\n=== API Key Checklist for APEX-ULTRA™ Revenue Streams ===\n")
    for key in REQUIRED_KEYS:
        print(f"- {key['name']}: {key['description']}")
        print(f"  Where to get: {key['where_to_get']}")
        print(f"  Example .env entry: {key['example']}\n")
    print("Best Practices:")
    print("- Store all API keys in a .env file at the project root.")
    print("- NEVER commit your .env file to version control. Add '.env' to your .gitignore.")
    print("- Provide a .env.example file (no secrets) for other developers.")
    print("- Use the python-dotenv package to load environment variables securely.")
    print("- Rotate and revoke API keys if you suspect they are compromised.")
    print("- For more info, see: https://dev.to/hamznabil/secure-api-key-handling-in-python-projects-1kg7\n")

def check_env_keys():
    load_dotenv()
    missing = []
    for key in REQUIRED_KEYS:
        if not os.environ.get(key["name"]):
            missing.append(key["name"])
    if missing:
        print("\nWARNING: The following API keys are missing from your .env file:")
        for k in missing:
            print(f"- {k}")
        print("\nPlease add them to your .env file before running the bot for full functionality.\n")
    else:
        print("\nAll required API keys are present in your .env file.\n")

if __name__ == "__main__":
    print_api_key_checklist()
    check_env_keys() 