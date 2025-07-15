import { NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"

const execAsync = promisify(exec)

export async function POST() {
  try {
    // Run the Python script to update model predictions
    await execAsync("python scripts/climate_model.py")

    return NextResponse.json({
      success: true,
      message: "Model updated successfully",
    })
  } catch (error) {
    console.error("Error updating model:", error)
    return NextResponse.json(
      {
        success: false,
        error: "Failed to update model",
      },
      { status: 500 },
    )
  }
}
