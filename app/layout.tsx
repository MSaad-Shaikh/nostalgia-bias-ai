import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], weight: ['300', '400', '500', '600', '700'] })

export const metadata: Metadata = {
  title: 'AI Bias Inspector — Unbiased AI Decisions',
  description: 'Ensuring fairness and detecting bias in automated decisions. Inspect datasets for hidden unfairness or discrimination.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-[#06080f] antialiased`}>{children}</body>
    </html>
  )
}
