'use client'

import { ChangeEvent, DragEvent, useRef, useState } from 'react'

type ScoreKey = 'Age Group' | 'Gender' | 'Ethnicity'

const SCORE_LABELS: ScoreKey[] = ['Age Group', 'Gender', 'Ethnicity']

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null)
  const [csvColumns, setCsvColumns] = useState<string[]>([])
  const [targetColumn, setTargetColumn] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [analysisState, setAnalysisState] = useState<
    'idle' | 'analyzing' | 'complete'
  >('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [aiReport, setAiReport] = useState<string>('')
  const [scores, setScores] = useState<Record<ScoreKey, number>>({
    'Age Group': 0.92,
    Gender: 0.58,
    Ethnicity: 0.64,
  })
  const fileInputRef = useRef<HTMLInputElement>(null)
  const loadCsvHeaders = async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch('http://127.0.0.1:5000/api/headers', {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch CSV headers (${response.status}).`)
    }

    const data = await response.json()
    const columns = Array.isArray(data?.columns)
      ? data.columns.map((column: unknown) => String(column))
      : []

    if (columns.length === 0) {
      throw new Error('No columns found in uploaded CSV.')
    }

    setCsvColumns(columns)
    setTargetColumn(columns[0])
  }

  const handleCsvFile = async (file: File) => {
    setSelectedFile(file)
    setSelectedFileName(file.name)
    setErrorMessage(null)
    setAnalysisState('idle')
    setAiReport('')

    try {
      await loadCsvHeaders(file)
    } catch (error) {
      setCsvColumns([])
      setTargetColumn('')
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Failed to read CSV headers from backend.',
      )
    }
  }

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)

    const file = event.dataTransfer.files[0]
    if (file && file.name.toLowerCase().endsWith('.csv')) {
      void handleCsvFile(file)
    }
  }

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      void handleCsvFile(file)
    }
  }

  const analyzeData = async () => {
    if (!selectedFile) {
      setErrorMessage('Please upload a CSV file first.')
      return
    }

    setErrorMessage(null)
    setAnalysisState('analyzing')
    setAiReport('')
    const formData = new FormData()
    formData.append('file', selectedFile)
    if (targetColumn) {
      formData.append('target_column', targetColumn)
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/api/analyze', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json()
      const apiScores = data?.disparate_impact_scores

      if (!apiScores) {
        throw new Error('No disparate_impact_scores found in API response.')
      }

      setScores({
        'Age Group': Number(apiScores['Age Group']),
        Gender: Number(apiScores['Gender']),
        Ethnicity: Number(apiScores['Ethnicity']),
      })
      if (typeof data?.ai_report === 'string') {
        setAiReport(data.ai_report)
      }
      if (Array.isArray(data?.columns)) {
        const columns = data.columns.map((column: unknown) => String(column))
        setCsvColumns(columns)
      }
      if (typeof data?.target_column === 'string') {
        setTargetColumn(data.target_column)
      }
      setAnalysisState('complete')
    } catch (error) {
      setAnalysisState('idle')
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Failed to analyze the uploaded CSV file.',
      )
    }
  }

  return (
    <main className="min-h-screen bg-[#fafafa] text-[#0a0a0a]">
      <div className="mx-auto max-w-4xl px-6 py-12">
        <header className="mb-12 text-center">
          <div className="mb-4 inline-flex items-center gap-3">
            <svg
              className="h-10 w-10 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
              />
            </svg>
            <h1 className="text-3xl font-semibold tracking-tight">
              AI Bias Inspector
            </h1>
          </div>
          <p className="text-lg text-[#737373]">Data Fairness Analysis Tool</p>
        </header>

        <section className="mb-8">
          <div
            role="button"
            tabIndex={0}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                fileInputRef.current?.click()
              }
            }}
            className={`cursor-pointer rounded-xl border-2 border-dashed p-12 text-center transition-all duration-200 ${
              isDragging
                ? 'border-blue-600 bg-blue-50'
                : 'border-[#e5e5e5] hover:border-blue-600 hover:bg-blue-50/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileSelect}
            />
            <svg
              className="mx-auto mb-4 h-12 w-12 text-[#737373]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="mb-2 font-medium">
              {selectedFileName ? (
                <>
                  <span className="text-blue-600">{selectedFileName}</span>{' '}
                  selected
                </>
              ) : (
                'Drop your CSV file here'
              )}
            </p>
            <p className="text-sm text-[#737373]">or click to browse</p>
          </div>
        </section>

        <section className="mb-8">
          <label htmlFor="targetColumn" className="mb-2 block text-sm font-medium">
            Target Outcome Column
          </label>
          <select
            id="targetColumn"
            value={targetColumn}
            onChange={(event) => setTargetColumn(event.target.value)}
            className="w-full rounded-lg border border-[#e5e5e5] bg-white px-4 py-3 text-[#0a0a0a] transition-all focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-600"
          >
            {csvColumns.length === 0 ? (
              <option value="" disabled>
                Upload a CSV to load columns...
              </option>
            ) : (
              csvColumns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))
            )}
          </select>
        </section>

        <section className="mb-12">
          <button
            type="button"
            onClick={analyzeData}
            disabled={analysisState === 'analyzing'}
            className={`flex w-full items-center justify-center gap-2 rounded-lg px-6 py-4 font-medium text-white transition-all duration-150 active:scale-[0.99] ${
              analysisState === 'complete'
                ? 'bg-emerald-600'
                : 'bg-blue-600 hover:bg-blue-700'
            } disabled:cursor-not-allowed disabled:opacity-90`}
          >
            {analysisState === 'analyzing' ? (
              <>
                <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                Analyzing...
              </>
            ) : analysisState === 'complete' ? (
              <>
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M5 13l4 4L19 7"
                  />
                </svg>
                Analysis Complete
              </>
            ) : (
              <>
                <svg
                  className="h-5 w-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                Analyze Data
              </>
            )}
          </button>
          {errorMessage && (
            <p className="mt-3 text-sm text-red-600" role="alert">
              {errorMessage}
            </p>
          )}
        </section>

        <section>
          <h2 className="mb-6 flex items-center gap-2 text-xl font-semibold">
            <svg
              className="h-5 w-5 text-[#737373]"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Results: Disparate Impact Scores
          </h2>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {SCORE_LABELS.map((label) => {
              const score = scores[label]
              const isFair = score >= 0.8
              const progressWidth = `${Math.round(score * 100)}%`

              return (
                <div
                  key={label}
                  className={`rounded-xl border bg-white p-6 shadow-sm ${
                    isFair ? 'border-emerald-200' : 'border-red-200'
                  }`}
                >
                  <div className="mb-4 flex items-center justify-between">
                    <span
                      className={`rounded-full px-3 py-1 text-sm font-medium ${
                        isFair
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {isFair ? 'Fair' : 'High Bias Warning'}
                    </span>
                    {isFair ? (
                      <svg
                        className="h-6 w-6 text-emerald-500"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="h-6 w-6 text-red-500"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                        />
                      </svg>
                    )}
                  </div>
                  <h3 className="mb-1 text-sm font-medium text-[#737373]">{label}</h3>
                  <p className="mb-2 text-3xl font-semibold">{score.toFixed(2)}</p>
                  <div
                    className={`h-2 w-full rounded-full ${
                      isFair ? 'bg-emerald-100' : 'bg-red-100'
                    }`}
                  >
                    <div
                      className={`h-2 rounded-full ${
                        isFair ? 'bg-emerald-500' : 'bg-red-500'
                      }`}
                      style={{ width: progressWidth }}
                    />
                  </div>
                  <p className="mt-2 text-xs text-[#737373]">
                    {isFair
                      ? 'Within acceptable range (0.8 - 1.25)'
                      : 'Below threshold - review required'}
                  </p>
                </div>
              )
            })}
          </div>
        </section>

        <section className="mt-8">
          <div className="w-full rounded-2xl border border-[#e5e5e5] bg-white p-6 shadow-sm md:p-8">
            <h3 className="text-xl font-semibold tracking-tight">
              AI Mitigation Strategy
            </h3>

            {analysisState === 'analyzing' ? (
              <div className="mt-5 space-y-3" aria-live="polite" aria-busy="true">
                <div className="h-4 w-full animate-pulse rounded bg-neutral-200" />
                <div className="h-4 w-[95%] animate-pulse rounded bg-neutral-200" />
                <div className="h-4 w-[92%] animate-pulse rounded bg-neutral-200" />
                <div className="h-4 w-[88%] animate-pulse rounded bg-neutral-200" />
                <div className="h-4 w-[90%] animate-pulse rounded bg-neutral-200" />
              </div>
            ) : aiReport ? (
              <p className="mt-4 whitespace-pre-line text-[15px] leading-7 text-[#1f2937]">
                {aiReport}
              </p>
            ) : (
              <p className="mt-4 text-sm text-[#737373]">
                Run an analysis to generate a mitigation strategy tailored to your
                uploaded dataset.
              </p>
            )}
          </div>
        </section>

        <footer className="mt-16 border-t border-[#e5e5e5] pt-8 text-center text-sm text-[#737373]">
          <p>
            AI Bias Inspector helps identify potential fairness issues in machine
            learning datasets.
          </p>
        </footer>
      </div>
    </main>
  )
}
