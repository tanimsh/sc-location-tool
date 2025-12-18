# South Carolina Facility Location Analysis Tool

Interactive web application for optimal facility location selection using Maximum Coverage Location-Allocation (MCLP) analysis.

## 🎯 Live Demo

🔗 **[Launch Application](https://YOUR-APP-URL.streamlit.app)** *(Update after deployment)*

## ✨ Features

- **Maximum Coverage Optimization** - Select optimal facility locations to maximize population coverage
- **Road Network Analysis** - Calculate actual travel times using real road networks
- **Service Area Visualization** - Interactive maps showing reachable areas (isochrones)
- **Ranked Recommendations** - Facilities ranked by coverage impact
- **Multiple Facility Types** - Churches, hospitals, clinics, food banks, and more
- **Export Results** - Download selected sites as CSV or GeoJSON

## 🚀 Quick Start

### Online (Recommended)
Visit the live application: [YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/sc-location-tool.git
cd sc-location-tool

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 📊 Usage

1. **Select ZIP Code** - Choose your area of interest
2. **Choose Facility Types** - Select which types of facilities to consider
3. **Set Parameters** - Configure travel mode, time threshold, number of facilities
4. **Run Analysis** - Click "Run Analysis" to optimize locations
5. **View Results** - See proposed sites on interactive map with service areas

## 🗺️ Analysis Methods

### Manhattan Distance (Fast)
- Grid-based distance calculation
- ~30 mph average speed for driving
- Suitable for initial screening

### Road Network Analysis (Accurate)
- Downloads actual road network from OpenStreetMap
- Calculates real driving/walking routes
- Considers road types and speed limits
- More accurate but slower

## 📈 Output

The tool provides:
- **Interactive Map** with selected facilities and service areas
- **Ranked Table** showing facilities by coverage impact
- **Coverage Statistics** - Total population covered, percentage, demand points
- **Export Options** - CSV and GeoJSON formats

## 🛠️ Technology Stack

- **Streamlit** - Web application framework
- **PuLP** - Linear programming optimization
- **OSMnx** - Road network analysis
- **Folium** - Interactive mapping
- **GeoPandas** - Spatial data processing

## 📄 Methodology

This tool implements the Maximum Coverage Location Problem (MCLP):
- **Objective:** Maximize covered population
- **Constraint:** Select exactly N facilities
- **Coverage:** Population point is covered if within travel time threshold
- **Ranking:** Facilities ranked by individual coverage contribution

## 📝 Citation

If you use this tool in your research, please cite:

```
[Your Name], [Year]. South Carolina Facility Location Analysis Tool. 
GitHub repository: https://github.com/YOUR-USERNAME/sc-location-tool
```

## 👤 Author

**Shakhawat H. Tanim**
- Affiliation: Clemson University 
- Website: (https://stanim.people.clemson.edu/index.html)

## 📜 License

[Choose your license: MIT, Apache 2.0, etc.]

## 🙏 Acknowledgments

- Clemson University
- [Any funding sources]
- [Any collaborators]

## 🐛 Issues & Feedback

Found a bug? Have a suggestion? 
[Open an issue](https://github.com/YOUR-USERNAME/sc-location-tool/issues)

---

**Last Updated:** December 2024
