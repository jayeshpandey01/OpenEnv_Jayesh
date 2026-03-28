class Environment(ABC, Generic[ActT, ObsT, StateT]):
    """Base class for all environment servers following Gym/Gymnasium API.

    Args:
        transform: Optional transform to apply to observations
        rubric: Optional rubric for reward computation. When provided, the
            rubric's output can be used to set the observation's reward in step().

    Class Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Whether this environment supports concurrent sessions.
            When True, multiple WebSocket connections can each have their own
            environment instance (up to max_concurrent_envs). When False (default),
            the environment should only be used with a single session at a time.

            Set this to True in your Environment subclass if:
            - The environment uses proper session isolation (e.g., unique working dirs)
            - No shared mutable state exists between instances
            - External resources (databases, APIs) can handle concurrent access

    Attributes:
        rubric: Optional rubric for computing rewards. Environments can set this
            in __init__ and use it in step() to compute observation rewards.
            Training infrastructure can access it for introspection:
                for name, r in env.rubric.named_rubrics():
                    print(f"{name}: {r.last_score}")

    See RFC 004 for rubric design: rfcs/004-rubrics.md
    """

    # Class-level flag indicating whether this environment supports concurrent sessions
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    # Optional rubric for reward computation
    rubric: Optional["Rubric"]

    def __init__(
        self,
        transform: Optional[Transform[ObsT]] = None,
        rubric: Optional["Rubric"] = None,
    ):
        self.transform = transform
        self.rubric = rubric

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Reset the environment and return initial observation."""
        pass

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Async version of reset. Default implementation calls sync reset.

        Override to provide true async implementation.
        """
        return self.reset(seed=seed, episode_id=episode_id, **kwargs)

    @abstractmethod
    def step(
        self,
        action: ActT,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Take a step in the environment."""
        pass

    async def step_async(
        self,
        action: ActT,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ObsT:
        """Async version of step. Default implementation calls sync step.

        Override to provide true async implementation.
        """
        return self.step(action, timeout_s=timeout_s, **kwargs)

    @property
    @abstractmethod
    def state(self) -> StateT:
        """Get the current environment state."""
        pass

    def get_metadata(self) -> EnvironmentMetadata:
        """
        Get metadata about this environment.

        Override this method to provide custom metadata for the environment.
        Default implementation returns basic metadata derived from class name.

        Returns:
            EnvironmentMetadata with environment information
        """
        return EnvironmentMetadata(
            name=self.__class__.__name__,
            description=f"{self.__class__.__name__} environment",
            version="1.0.0",
        )

    def _apply_transform(self, observation: ObsT) -> ObsT:
        """Apply transform if one is provided."""
        if self.transform is not None:
            return self.transform(observation)
        return observation

    def _apply_rubric(self, action: ActT, observation: ObsT) -> float:
        """Apply rubric if one is provided.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            Reward value from the rubric, or 0.0 if no rubric is set.

        Usage in step():
            def step(self, action: MyAction, ...) -> MyObservation:
                # ... execute action and create observation ...
                observation.reward = self._apply_rubric(action, observation)
                return observation
        """
        if self.rubric is not None:
            return self.rubric(action, observation)
        return 0.0

    async def _apply_rubric_async(self, action: ActT, observation: ObsT) -> float:
        """Apply rubric asynchronously if one is provided.

        Args:
            action: The action taken by the agent.
            observation: The resulting observation.

        Returns:
            Reward value from the rubric, or 0.0 if no rubric is set.

        Usage in step_async():
            async def step_async(self, action: MyAction, ...) -> MyObservation:
                # ... execute action and create observation ...
                observation.reward = await self._apply_rubric_async(action, observation)
                return observation
        """
        if self.rubric is not None:
            result = self.rubric(action, observation)
            # If rubric returns a coroutine, await it
            if inspect.iscoroutine(result):
                return await result
            return result
        return 0.0

    def _reset_rubric(self) -> None:
        """Reset the rubric state if one is provided.

        Call this in reset() to clear any trajectory state in the rubric.

        Usage in reset():
            def reset(self, ...) -> MyObservation:
                self._reset_rubric()
                # ... create initial observation ...
                return observation
        """
        if self.rubric is not None:
            self.rubric.reset()

    async def _reset_rubric_async(self) -> None:
        """Reset the rubric state asynchronously if one is provided.

        Call this in reset_async() to clear any trajectory state in the rubric.

        Usage in reset_async():
            async def reset_async(self, ...) -> MyObservation:
                await self._reset_rubric_async()
                # ... create initial observation ...
                return observation
        """
        if self.rubric is not None:
            # Check if rubric has async reset method
            if hasattr(self.rubric, "reset_async"):
                result = self.rubric.reset_async()
                if inspect.iscoroutine(result):
                    await result
            else:
                self.rubric.reset()

    def close(self) -> None:
        """Clean up resources used by the environment.

        Override this method to implement custom cleanup logic.
        Called when the environment is being destroyed or reset.
        """
        pass

