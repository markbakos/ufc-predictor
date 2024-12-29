<script setup lang="ts">
import { computed, onMounted } from 'vue'

const props = defineProps<{winner: string, confidence: number, advantages: Record<string, number>}>()

const maxValue = computed(() => {
  return Math.max(...Object.values(props.advantages).map(Math.abs))
})

function getBarWidth(value: number) {
  const percentage = (Math.abs(value) / maxValue.value) * 50
  return `${percentage}%`
}

function getBarPosition(value: number) {
  return value >= 0 ? 'left-1/2' : 'right-1/2'
}

onMounted(() => {
  const bars = document.querySelectorAll('.advantage-bar')
  bars.forEach((bar, index) => {
    setTimeout(() => {
      bar.classList.add('animate-bar')
    }, index * 100)
  })
})
</script>

<template>
  <div class="text-center">
    <h2 class="text-3xl font-bold text-center text-gray-100 mb-6">Prediction Results</h2>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
      <div class="bg-gray-800 p-6 rounded-lg shadow-md">
        <p class="text-xl text-gray-300 mb-2">Winner:</p>
        <p class="text-3xl font-bold text-gray-100">{{ winner }}</p>
      </div>
      <div class="bg-gray-800 p-6 rounded-lg shadow-md">
        <p class="text-xl text-gray-300 mb-2">Confidence:</p>
        <p class="text-3xl font-bold text-gray-100">{{ (confidence*100).toFixed(2) }}%</p>
        <p class="text-sm text-gray-400">({{ confidence }})</p>
      </div>
    </div>
  </div>
  <h3 class="text-2xl font-semibold text-gray-100 mb-6">Advantages:</h3>
  <div class="space-y-6">
    <div v-for="(value, key) in advantages" :key="key" class="bg-gray-800 p-6 rounded-lg shadow-lg hover:bg-gray-700 transition-all duration-300">
      <div class="flex justify-between items-center mb-3">
        <span class="text-lg font-medium text-gray-300">{{ key }}:</span>
        <span class="text-lg font-bold" :class="value >= 0 ? 'text-green-400' : 'text-red-400'">
          {{ value.toFixed(2) }}
        </span>
      </div>
      <div class="relative h-8 bg-gray-900 rounded-full overflow-hidden">
        <div class="absolute top-0 bottom-0 w-px bg-gray-600 left-1/2"></div>
        <div
            class="advantage-bar absolute top-0 bottom-0 opacity-0"
            :class="[getBarPosition(value), value >= 0 ? 'bg-green-500' : 'bg-red-500']"
            :style="{ width: getBarWidth(value) }"
        ></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
@keyframes barAnimation {
  0% { opacity: 0; transform: scaleX(0); }
  100% { opacity: 1; transform: scaleX(1); }
}

.animate-bar {
  animation: barAnimation 0.5s ease-out forwards;
  transform-origin: center;
}
</style>
