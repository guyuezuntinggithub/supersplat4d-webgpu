enum State {
    selected = 1,
    locked = 2,
    deleted = 4,
    /** 仅用于 unified 动态：当前帧不显示的 splat（写入 state 纹理，不修改持久 state 数组） */
    frameInactive = 8
}

export { State };
