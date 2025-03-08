//! A non growable, dynamically allocated collection

use core::alloc::Layout;
use core::mem::{MaybeUninit, replace, transmute};
use core::ops::{Index, IndexMut};
use core::ptr::{NonNull, slice_from_raw_parts_mut};
use core::slice::{Iter, IterMut};

extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::alloc::{alloc, dealloc, handle_alloc_error};

#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc, handle_alloc_error};

// use num_traits::Zero;

#[derive(Debug)]
pub struct ArrayPtr<T>(NonNull<[T]>);
impl<T> Drop for ArrayPtr<T> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(self.as_slice().len()).expect("A sane layout");
        unsafe { dealloc(self.0.as_ptr().cast(), layout) };
    }
}
unsafe impl<T> Send for ArrayPtr<T> {}
unsafe impl<T> Sync for ArrayPtr<T> {}
impl<T> Index<usize> for ArrayPtr<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        let s = unsafe { self.0.as_ref() };
        &s[index]
    }
}
impl<T> IndexMut<usize> for ArrayPtr<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let s = unsafe { self.0.as_mut() };
        &mut s[index]
    }
}
impl<T> ArrayPtr<T> {
    pub fn new_uninit(count: usize) -> ArrayPtr<MaybeUninit<T>> {
        let layout = Layout::array::<T>(count).expect("A sane layout");

        let ptr = unsafe { alloc(layout) } as *mut MaybeUninit<T>;

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        let ptr_slice = slice_from_raw_parts_mut(ptr, count);

        // SAFETY: We validated above the pointer is non-null, and handle allocation errors accordingly
        let inner = unsafe { NonNull::new_unchecked(ptr_slice) };

        ArrayPtr(inner)
    }
    pub fn shift_once(&mut self)
    where
        T: Copy,
    {
        let fst = self[0];
        let bound = self.len() - 1;
        for i in 0..bound {
            self[i] = self[i + 1]
        }
        self[bound] = fst;
    }
    pub fn from_raw_parts(data: *mut T, len: usize) -> ArrayPtr<T> {
        let ptr = core::ptr::slice_from_raw_parts_mut(data, len);
        assert!(!ptr.is_null());
        let inner = unsafe { NonNull::new_unchecked(ptr) };
        ArrayPtr(inner)
    }
}
impl<T> ArrayPtr<MaybeUninit<T>> {
    /// # Safety
    /// Caller must guarantee each member of the
    /// [`ArrayPtr`] has been written to with a
    /// valid instance of `T`
    pub unsafe fn assume_init(self) -> ArrayPtr<T> {
        debug_assert!(!self.is_empty(), "Calling assume_init on an empty array!");
        unsafe { transmute(self) }
    }
}

impl<T> ArrayPtr<T>
where
    T: num_traits::Zero,
{
    pub fn new_zeroed(count: usize) -> ArrayPtr<T> {
        let mut arr = ArrayPtr::new_uninit(count);
        for val in arr.iter_mut() {
            val.write(T::zero());
        }
        unsafe { arr.assume_init() }
    }
}
impl<T> ArrayPtr<T>
where
    T: Default,
{
    pub fn new_default(count: usize) -> ArrayPtr<T> {
        let mut arr = ArrayPtr::new_uninit(count);
        for val in arr.iter_mut() {
            val.write(T::default());
        }
        unsafe { arr.assume_init() }
    }
}
impl<T> AsRef<[T]> for ArrayPtr<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        unsafe { self.0.as_ref() }
    }
}
impl<T> AsMut<[T]> for ArrayPtr<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { self.0.as_mut() }
    }
}
impl<T> ArrayPtr<T> {
    pub fn from_fn(count: usize, f: impl Fn(usize) -> T) -> ArrayPtr<T> {
        let mut buf = ArrayPtr::new_uninit(count);
        for (idx, elt) in buf.iter_mut().enumerate() {
            elt.write(f(idx));
        }
        unsafe { buf.assume_init() }
    }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, T> {
        self.as_slice().iter()
    }
    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}
impl<T> FromIterator<T> for ArrayPtr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let v: Vec<T> = iter.into_iter().collect();

        let mut arr = ArrayPtr::<T>::new_uninit(v.len());
        for (idx, val) in v.into_iter().enumerate() {
            arr[idx].write(val);
        }

        unsafe { arr.assume_init() }
    }
}
pub struct RingPtr<T> {
    inner: NonNull<[T]>,
    head: usize,
    tail: usize,
}
impl<T> Drop for RingPtr<T> {
    fn drop(&mut self) {
        let capacity = self.capacity();
        if capacity == 0 {
            return;
        }
        let layout = Layout::array::<T>(capacity).expect("A sane layout");
        unsafe { dealloc(self.inner.as_ptr().cast(), layout) };
    }
}

unsafe impl<T> Send for RingPtr<T> {}
unsafe impl<T> Sync for RingPtr<T> {}
impl<T> Index<usize> for RingPtr<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}
impl<T> IndexMut<usize> for RingPtr<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}
impl<T> RingPtr<T> {
    pub fn new_uninit(count: usize) -> RingPtr<MaybeUninit<T>> {
        let layout = Layout::array::<T>(count).expect("A sane layout");

        let ptr = unsafe { alloc(layout) } as *mut MaybeUninit<T>;

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        let ptr_slice = slice_from_raw_parts_mut(ptr, count);

        // SAFETY: We validated above the pointer is non-null, and handle allocation errors accordingly
        let inner = unsafe { NonNull::new_unchecked(ptr_slice) };

        RingPtr {
            inner,
            head: 0,
            tail: 0,
        }
    }
}

impl<T> RingPtr<MaybeUninit<T>> {
    /// # Safety
    /// Caller must guarantee each member of the
    /// [`RingPtr`] has been written to with a
    /// valid instance of `T`
    pub unsafe fn assume_init(self) -> RingPtr<T> {
        debug_assert!(
            self.capacity() != 0,
            "Calling assume_init on an empty array!"
        );
        unsafe { transmute(self) }
    }
}
#[cfg(feature = "num-traits")]
impl<T> RingPtr<T>
where
    T: num_traits::Zero,
{
    pub fn new_zeroed(count: usize) -> RingPtr<T> {
        let mut arr = RingPtr::new_uninit(count);
        for val in arr.iter_mut() {
            val.write(T::zero());
        }
        unsafe { arr.assume_init() }
    }
}
impl<T> RingPtr<T>
where
    T: Default,
{
    pub fn new_default(count: usize) -> RingPtr<T> {
        let mut arr = RingPtr::new_uninit(count);
        for val in arr.iter_mut() {
            val.write(T::default());
        }
        unsafe { arr.assume_init() }
    }
}
pub type Pair<T> = (T, T);
impl<T> RingPtr<T> {
    #[inline(always)]
    fn at(&mut self, offset: usize) -> *mut T {
        unsafe { self.as_mut_slice().as_mut_ptr().add(offset) }
    }
    /// Returns true if the capacity is zero or the current tail is the next head position
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty() || self.tail == self.next_head_pos()
    }
    /// Returns the 'count' of valid elements in the collection, not the capacity
    #[inline(always)]
    pub const fn len(&self) -> usize {
        if self.head < self.tail {
            self.head - self.tail
        } else {
            let from_head = self.capacity() - self.head;
            from_head + self.tail
        }
    }
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.inner.len()
    }
    #[inline(always)]
    const fn next_head_pos(&self) -> usize {
        (self.head + 1) % self.capacity()
    }
    #[inline(always)]
    const fn next_tail_pos(&self) -> usize {
        (self.tail + 1) % self.capacity()
    }

    pub fn push(&mut self, val: T) {
        unsafe { self.at(self.tail).write(val) };
        self.tail = self.next_tail_pos();
    }
    pub fn replace(&mut self, src: T) -> T {
        unsafe { replace(&mut *self.at(self.tail), src) }
    }
    pub fn reset(&mut self) {
        self.head = 0;
        self.tail = 0;
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let val = unsafe { self.at(self.head).read() };
            self.head = self.next_head_pos();
            Some(val)
        }
    }
    pub fn peek(&self) -> &T {
        &self.as_slice()[self.head]
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.inner.as_ref() }
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { self.inner.as_mut() }
    }
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, T> {
        self.as_slice().iter()
    }
    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }
}
pub struct CircularBuffer<T> {
    buffer: ArrayPtr<T>,
    cursor: usize,
}
impl<T> From<ArrayPtr<T>> for CircularBuffer<T> {
    fn from(value: ArrayPtr<T>) -> Self {
        debug_assert!(
            !value.is_empty(),
            "Making circular buffer from empty ArrayPtr!"
        );
        CircularBuffer {
            buffer: value,
            cursor: 0,
        }
    }
}
#[cfg(feature = "num-traits")]
impl<T> CircularBuffer<T> {
    pub fn new_zeroed(count: usize) -> CircularBuffer<T>
    where
        T: num_traits::Zero,
    {
        ArrayPtr::new_zeroed(count).into()
    }
}
impl<T> CircularBuffer<T> {
    pub fn new_default(count: usize) -> CircularBuffer<T>
    where
        T: Default,
    {
        ArrayPtr::new_default(count).into()
    }
    /// Returns what the next cursor value would be
    fn incremented_cursor(&self) -> usize {
        (self.cursor + 1) % (self.buffer.len())
    }
    fn decremented_cursor(&self) -> usize {
        if self.cursor == 0 {
            self.buffer.len() - 1
        } else {
            (self.cursor - 1) % (self.buffer.len())
        }
    }
    pub fn current_as_ref(&self) -> &T {
        unsafe { self.buffer.as_slice().get_unchecked(self.cursor) }
    }
    fn current_as_mut(&mut self) -> &mut T {
        unsafe { self.buffer.as_mut_slice().get_unchecked_mut(self.cursor) }
    }
    /// Returns the previous value, and increments cursor
    #[inline(always)]
    pub fn update_forwards(&mut self, src: T) -> (T, T)
    where
        T: Copy,
    {
        let prev = *self.current_as_ref();
        self.cursor = self.incremented_cursor();
        let curr = replace(self.current_as_mut(), src);
        (prev, curr)
    }
    pub fn update_backwards(&mut self, src: T) -> (T, T)
    where
        T: Copy,
    {
        let prev = *self.current_as_ref();
        self.cursor = self.decremented_cursor();
        let curr = replace(self.current_as_mut(), src);
        (prev, curr)
    }
    #[inline(always)]
    pub const fn set_cursor(&mut self, val: usize) {
        self.cursor = val;
    }
    #[inline(always)]
    pub const fn reset_cursor(&mut self) {
        self.set_cursor(0);
    }
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_slice()
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer.as_mut_slice()
    }
}
impl<T> Index<usize> for CircularBuffer<T> {
    type Output = <ArrayPtr<T> as Index<usize>>::Output;
    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}
impl<T> IndexMut<usize> for CircularBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buffer[index]
    }
}

#[cfg(feature = "circular-buffer-fft")]
impl<T> CircularBuffer<num_complex::Complex<T>> {
    #[inline(always)]
    pub fn inplace_fourier_transform(
        &mut self,
        transformer: &dyn rustfft::Fft<T>,
        scratch: &mut [num_complex::Complex<T>],
    ) where
        T: rustfft::FftNum,
    {
        transformer.process_with_scratch(self.buffer.as_mut_slice(), scratch);
    }
    pub fn inplace_hilbert_transform<'s>(
        &'s mut self,
        processor: &dyn rustfft::Fft<T>,
        reverse_processor: &dyn rustfft::Fft<T>,
        scratch: &mut [num_complex::Complex<T>],
    ) -> &'s [num_complex::Complex<T>]
    where
        T: rustfft::FftNum + num_traits::NumCast,
        num_complex::Complex<T>: core::ops::MulAssign + core::ops::DivAssign<T>,
    {
        self.inplace_fourier_transform(processor, scratch);

        self.apply_hilbert_filter();

        self.inplace_fourier_transform(reverse_processor, scratch);
        // Ensure properly scaled values following IFFT
        self.batch_downscale();

        self.as_slice()
    }
    #[inline(always)]
    pub fn apply_hilbert_filter(&mut self)
    where
        T: rustfft::FftNum + num_traits::NumCast + num_traits::Zero,
        num_complex::Complex<T>: core::ops::MulAssign,
    {
        use num_complex::Complex;

        let n = self.buffer.as_slice().len();
        for (i, elt) in self.buffer.iter_mut().enumerate().take(n) {
            let multiplier = match i {
                0 => <num_complex::Complex<T> as num_traits::Zero>::zero(), // Zero DC
                i if i < n / 2 => Complex::new(T::zero(), -T::one()), // -i for positive frequencies
                _ => <num_complex::Complex<T> as num_traits::Zero>::zero(), // Zero negative frequencies
            };
            *elt *= multiplier;
        }
    }
    #[inline(always)]
    fn batch_downscale(&mut self)
    where
        T: num_traits::Num + num_traits::NumCast + Copy,
        num_complex::Complex<T>: core::ops::DivAssign<T>,
    {
        let n = self.buffer.len();
        let n_t: T = num_traits::NumCast::from(n).expect("A usize for downscaling");
        for elt in self.buffer.iter_mut() {
            *elt /= n_t;
        }
    }
}
