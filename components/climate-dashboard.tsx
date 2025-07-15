"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts"
import { Download, TrendingUp, Thermometer, Calendar, Cloud, Waves, Droplets } from "lucide-react" // Added new icons

interface PredictionData {
  future_predictions: Array<{
    year: number
    predicted_temperature: number
    temperature_anomaly: number
    uncertainty: number
  }>
  monthly_data: Array<{
    month: string
    temperature: number
    month_num: number
  }>
  model_info: {
    train_score: number
    test_score: number
    mae: number
    rmse: number
    feature_count: number
    model_type: string
    data_source: string
  }
  sample_data: Array<any> // This will now contain the new features
  dataset_info: {
    total_records: number
    date_range: string
    features_used: string[]
    temperature_stats: {
      mean_temp: number
      std_temp: number
      min_temp: number
      max_temp: number
      "CO2 Emissions (Tons/Capita)"?: number
      "Sea Level Rise (mm)"?: number
      "Rainfall (mm)"?: number
      Population?: number
      "Renewable Energy (%)"?: number
      "Extreme Weather Events"?: number
      "Forest Area (%)"?: number
    }
  }
}

export default function ClimateDashboard() {
  const [data, setData] = useState<PredictionData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedScenario, setSelectedScenario] = useState("baseline")

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch("/climate-predictions.json")
        const jsonData = await response.json()
        setData(jsonData)
      } catch (error) {
        console.error("Error loading climate data:", error)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  const generateScenarioData = (scenario: string) => {
    if (!data) return data?.future_predictions || []

    return data.future_predictions.map((item) => {
      let adjustment = 0
      switch (scenario) {
        case "optimistic":
          adjustment = -0.5 // Cooler temperatures
          break
        case "pessimistic":
          adjustment = 1.5 // Warmer temperatures
          break
        default:
          adjustment = 0
      }

      return {
        ...item,
        predicted_temperature: Math.round((item.predicted_temperature + adjustment) * 100) / 100,
        temperature_anomaly: Math.round((item.temperature_anomaly + adjustment) * 100) / 100,
      }
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-lg">Loading Climate Model...</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <p className="text-lg text-red-600">Failed to load climate data</p>
          <Button onClick={() => window.location.reload()} className="mt-4">
            Retry
          </Button>
        </div>
      </div>
    )
  }

  const scenarioData = generateScenarioData(selectedScenario)
  const currentTemp = data.monthly_data[new Date().getMonth()]?.temperature || 0
  const futureTemp = scenarioData[scenarioData.length - 1]?.predicted_temperature || 0
  const tempChange = (futureTemp - currentTemp).toFixed(1)

  const { temperature_stats } = data.dataset_info

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-green-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Historical Climate Analysis Dashboard</h1>
          <p className="text-lg text-gray-600">Temperature Prediction Model using Historical Climate Data</p>
          <div className="flex justify-center gap-2 mt-4"> 
            <Badge variant="secondary">{data.dataset_info.date_range}</Badge>
            {/* <Badge variant="secondary">{data.dataset_info.total_records.toLocaleString()} Records</Badge> */}
            <Badge variant="secondary">ML Powered</Badge>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-green-600" />
                <div>
                  <p className="text-sm text-gray-600">Model Accuracy</p>
                  <p className="text-2xl font-bold text-green-600">{(data.model_info.test_score * 100).toFixed(1)}%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Thermometer className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-600">Current Month Temp</p>
                  <p className="text-2xl font-bold text-blue-600">{currentTemp}°C</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Calendar className="h-5 w-5 text-red-600" />
                <div>
                  <p className="text-sm text-gray-600">2030 Temp Projection</p>
                  <p className="text-2xl font-bold text-red-600">{futureTemp}°C</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Cloud className="h-5 w-5 text-gray-600" />
                <div>
                  <p className="text-sm text-gray-600">Avg CO2 Emissions</p>
                  <p className="text-2xl font-bold text-gray-800">
                    {temperature_stats["CO2 Emissions (Tons/Capita)"]?.toFixed(2) || "N/A"}
                    <span className="text-base font-normal"> T/Capita</span>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Waves className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm text-gray-600">Avg Sea Level Rise</p>
                  <p className="text-2xl font-bold text-blue-800">
                    {temperature_stats["Sea Level Rise (mm)"]?.toFixed(2) || "N/A"}
                    <span className="text-base font-normal"> mm</span>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center gap-2">
                <Droplets className="h-5 w-5 text-cyan-600" />
                <div>
                  <p className="text-sm text-gray-600">Avg Rainfall</p>
                  <p className="text-2xl font-bold text-cyan-800">
                    {temperature_stats["Rainfall (mm)"]?.toFixed(0) || "N/A"}
                    <span className="text-base font-normal"> mm</span>
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Charts */}
        <Tabs defaultValue="trends" className="space-y-4 p-2">
          <Card className="text-3xl">
            <TabsList className=" grid w-full grid-cols-3">
              <TabsTrigger className="text-2xl font-bold" value="trends">
                Trends
              </TabsTrigger>
              <TabsTrigger className="text-2xl font-bold" value="predictions">
                Future Predictions
              </TabsTrigger>
              {/* <TabsTrigger value="monthly">Monthly Analysis</TabsTrigger> */}
              <TabsTrigger className="text-2xl font-bold" value="model">
                Model Details
              </TabsTrigger>
            </TabsList>
          </Card>
          <TabsContent value="predictions" className="space-y-4">
            {/* Scenario Selection */}
            <Card>
              <CardHeader>
                <CardTitle>Climate Scenarios</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2 mb-4">
                  {[
                    { id: "optimistic", label: "Optimistic (-0.5°C)", color: "bg-green-100 text-green-800" },
                    { id: "baseline", label: "Baseline", color: "bg-blue-100 text-blue-800" },
                    { id: "pessimistic", label: "Pessimistic (+1.5°C)", color: "bg-red-100 text-red-800" },
                  ].map((scenario) => (
                    <Button
                      key={scenario.id}
                      variant={selectedScenario === scenario.id ? "default" : "outline"}
                      onClick={() => setSelectedScenario(scenario.id)}
                      className="flex-1"
                    >
                      {scenario.label}
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Temperature Predictions (2024-2030)</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={scenarioData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis label={{ value: "Temperature (°C)", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(value) => [`${value}°C`, "Predicted Temperature"]} />
                      <Line
                        type="monotone"
                        dataKey="predicted_temperature"
                        stroke="#8884d8"
                        strokeWidth={3}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Temperature Anomaly Trends</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={scenarioData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis label={{ value: "Anomaly (°C)", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(value) => [`${value}°C`, "Temperature Anomaly"]} />
                      <Line
                        type="monotone"
                        dataKey="temperature_anomaly"
                        stroke="#ff7300"
                        strokeWidth={3}
                        activeDot={{ r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="trends" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Dataset Overview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span>Total Records:</span>
                      <span className="font-mono">{data.dataset_info.total_records.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Date Range:</span>
                      <span className="font-mono">{data.dataset_info.date_range}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Data Source:</span>
                      <span className="font-mono">{data.model_info.data_source}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Features Used:</span>
                      <span className="font-mono">{data.model_info.feature_count}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* New Charts for Atmospheric Features */}
              <Card>
                <CardHeader>
                  <CardTitle>CO2 Emissions Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "CO2 (Tons/Capita)", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="CO2 Emissions (Tons/Capita)" stroke="#82ca9d" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Sea Level Rise Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "Sea Level (mm)", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Sea Level Rise (mm)" stroke="#8884d8" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Rainfall Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "Rainfall (mm)", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Rainfall (mm)" stroke="#00bcd4" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Population Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "Population", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Population" stroke="#ffc658" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Renewable Energy Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "Renewable Energy (%)", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Renewable Energy (%)" stroke="#4caf50" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Extreme Weather Events Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "Events", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Extreme Weather Events" stroke="#ff5722" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Forest Area Trend</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={data.sample_data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="Year" type="category" /> {/* Added type="category" */}
                      <YAxis label={{ value: "Forest Area (%)", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="Forest Area (%)" stroke="#795548" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="model" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Model Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span>Model Type:</span>
                      <Badge>{data.model_info.model_type}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Training R² Score:</span>
                      <span className="font-mono">{(data.model_info.train_score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Test R² Score:</span>
                      <span className="font-mono">{(data.model_info.test_score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Mean Absolute Error:</span>
                      <span className="font-mono">{data.model_info.mae.toFixed(4)}°C</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Root Mean Square Error:</span>
                      <span className="font-mono">{data.model_info.rmse.toFixed(4)}°C</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Download Models & Data</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <Button className="w-full" asChild>
                      <a href="/climate_model.pkl" download>
                        <Download className="mr-2 h-4 w-4" />
                        Download Climate Model (.pkl)
                      </a>
                    </Button>
                    <Button variant="outline" className="w-full bg-transparent" asChild>
                      <a href="/climate-predictions.json" download>
                        <Download className="mr-2 h-4 w-4" />
                        Download Predictions (.json)
                      </a>
                    </Button>
                    <Button variant="outline" className="w-full bg-transparent" asChild>
                      <a href="/climate_change_dataset.csv" download>
                        <Download className="mr-2 h-4 w-4" />
                        Download Dataset (.csv)
                      </a>
                    </Button>
                    <div className="text-sm text-gray-600 mt-4">
                      <p>
                        <strong>Features Used:</strong>
                      </p>
                      <ul className="list-disc list-inside text-xs space-y-1 mt-2">
                        {data.dataset_info.features_used.map((feature, index) => (
                          <li key={index}>{feature}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
