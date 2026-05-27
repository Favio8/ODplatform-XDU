import { useEffect, useRef } from "react";
import * as echarts from "echarts";

interface EChartProps {
  option: echarts.EChartsOption;
  className?: string;
  style?: React.CSSProperties;
}

export function EChart({ option, className, style }: EChartProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    let rafId: number | null = null;
    const init = () => {
      if (!chartRef.current) return;
      const chart = echarts.getInstanceByDom(chartRef.current);
      if (chart) {
        chart.setOption(option);
        chart.resize();
      } else {
        const newChart = echarts.init(chartRef.current);
        newChart.setOption(option);
      }
    };

    rafId = requestAnimationFrame(init);

    const observer = new ResizeObserver(() => {
      const chart = chartRef.current ? echarts.getInstanceByDom(chartRef.current) : null;
      chart?.resize();
    });
    if (chartRef.current) observer.observe(chartRef.current);

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      observer.disconnect();
      const chart = chartRef.current ? echarts.getInstanceByDom(chartRef.current) : null;
      chart?.dispose();
    };
  }, [option]);

  return <div ref={chartRef} className={className} style={style} />;
}
