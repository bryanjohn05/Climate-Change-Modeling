import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Climate Change Modeling',
  description: 'Bryan Sohan John',
 
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html suppressHydrationWarning lang="en">
      <body >{children}</body>
    </html>
  )
}
