# TikTok Doctor Content Scraper

A Streamlit application that uses the Apify TikTok scraper to find and analyze content from medical professionals on TikTok.

## Features

- 🔍 Search TikTok for doctor-related content
- 📊 Analyze engagement metrics (likes, views, shares, comments)
- ✅ Filter by verified accounts
- 📱 User-friendly Streamlit interface
- 📥 Export results to CSV
- 🩺 Focused on medical professional content

## Prerequisites

- Python 3.7+
- Apify account and API token
- Internet connection

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tiktok-doctor-scraper.git
cd tiktok-doctor-scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Get your Apify API token:
   - Go to [Apify](https://apify.com)
   - Sign up or log in
   - Navigate to Account → Integrations
   - Copy your API token

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your Apify API token in the sidebar

3. Configure your search parameters:
   - Add search queries (one per line)
   - Set maximum results per query
   - Choose search type (hashtag, user, keyword)

4. Click "Start Scraping" to begin

5. View and filter results

6. Download results as CSV

## Search Query Examples

For doctor-related content, try these queries:
- `doctor`
- `medical advice`
- `physician`
- `healthcare`
- `#doctor`
- `#medicalstudent`
- `#healthtips`
- `#medicalfacts`

## Configuration Options

### Search Parameters
- **Search Queries**: Keywords or hashtags to search for
- **Max Results**: Number of results per query (1-100)
- **Search Type**: hashtag, user, or keyword search

### Advanced Options
- **Include Engagement Metrics**: Shows likes, views, shares, comments
- **Verified Accounts Only**: Filter for verified medical professionals

## Data Fields

The scraper returns the following data for each TikTok post:
- Post ID and URL
- Text content
- Author information (name, verification status, followers)
- Engagement metrics (likes, shares, comments, views)
- Creation timestamp
- Hashtags used

## Rate Limits

- Apify has usage limits based on your plan
- The TikTok scraper respects TikTok's rate limits
- Consider running searches during off-peak hours

## Legal Considerations

- This tool is for research and educational purposes
- Respect TikTok's Terms of Service
- Be mindful of content creator rights
- Use data responsibly and ethically

## Troubleshooting

### Common Issues

1. **"Failed to start scraper"**: Check your API token
2. **No results found**: Try different search queries
3. **Timeout errors**: Reduce max results or try again later

### Support

- Check the [Apify TikTok Scraper documentation](https://apify.com/clockworks/tiktok-scraper)
- Review TikTok's current API limitations
- Ensure your Apify account has sufficient credits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is not affiliated with TikTok or Apify. Use responsibly and in accordance with all applicable terms of service and laws.
