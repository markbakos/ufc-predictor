<script setup lang="ts">
import { computed } from 'vue'

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
</script>

<template>
  <div class="text-center">
    <h2 class="text-2xl font-bold text-center text-gray-800">Prediction Results</h2>
    <p class="mt-4 text-center text-gray-700">Winner:</p>
    <p class="font-bold text-gray-700">{{winner}}</p>
    <p class="mt-2 text-gray-700">Confidence:</p>
    <p class="font-bold text-gray-700">{{(confidence*100).toFixed(2)}}%</p>
    <p class="text-gray-700">({{confidence}})</p>
  </div>
  <h3 class="text-2xl font-semibold text-gray-800 mb-4">Advantages:</h3>
  <div class="space-y-6">
    <div v-for="(value, key) in advantages" :key="key" class="bg-gray-50 p-4 rounded-lg">
      <div class="flex justify-between items-center mb-2">
        <span class="text-lg font-medium text-gray-700">{{ key }}:</span>
        <span class="text-lg font-bold" :class="value >= 0 ? 'text-green-600' : 'text-red-600'">
            {{ value.toFixed(2) }}
          </span>
      </div>
      <div class="relative h-6 bg-gray-200">
        <div class="absolute top-0 bottom-0 w-px bg-gray-400 left-1/2"></div>
        <div
            class="absolute top-0 bottom-0"
            :class="[getBarPosition(value), value >= 0 ? 'bg-green-500' : 'bg-red-500']"
            :style="{ width: getBarWidth(value) }"
        ></div>
      </div>
    </div>
  </div>
</template>
