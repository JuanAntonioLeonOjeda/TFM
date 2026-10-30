export function TipRow({ value, label, color }: { value: string; label: string; color?: string }) {
  return (
    <div className="tip-row">
      {color && <span className="tip-key" style={{ background: color }} />}
      <strong>{value}</strong> <span>{label}</span>
    </div>
  )
}
